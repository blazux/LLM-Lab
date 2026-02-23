import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# POSITIONAL ENCODINGS
# ============================================================================

class LearnedPositional(nn.Module):
    """Learned positional embeddings (like BERT/GPT-1)

    Simple trainable position embeddings. Classic baseline for comparing
    against modern methods like RoPE. Each position has a learned vector.
    """
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.position_embeddings(positions)


class SinusoidalPositional(nn.Module):
    """Sinusoidal positional encoding (original Transformer)"""
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:x.size(1), :]


class RoPE(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, d_k: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=d_k // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(d_k // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        """x: (batch, n_heads, seq_len, d_k)"""
        assert self.cos.size(0) >= x_BTHD.size(-2)
        cos = self.cos[None, None, :x_BTHD.size(-2), :]
        sin = self.sin[None, None, :x_BTHD.size(-2), :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_BTHD)


class ALiBi(nn.Module):
    """Attention with Linear Biases"""
    def __init__(self, n_heads: int, max_seq_len: int):
        super().__init__()
        slopes = torch.tensor(self._get_slopes(n_heads))
        positions = torch.arange(max_seq_len)
        bias = -slopes.unsqueeze(1) * positions.unsqueeze(0)
        self.register_buffer('bias', bias)

    @staticmethod
    def _get_slopes(n_heads: int):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            return (get_slopes_power_of_2(closest_power_of_2) +
                    ALiBi._get_slopes(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2])

    def get_bias(self, seq_len: int):
        """Returns bias for attention: (n_heads, seq_len, seq_len)"""
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return self.bias[:, :seq_len].unsqueeze(1) + causal_mask.to(self.bias.device)


class YARN(nn.Module):
    """Yet Another RoPE Extension (YaRN) - RoPE with extrapolation"""
    def __init__(self, d_k: int, max_seq_len: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=d_k // 4, dtype=torch.float32)
        angular_freq = angular_freq * scale
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(d_k // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        """x: (batch, n_heads, seq_len, d_k)"""
        assert self.cos.size(0) >= x_BTHD.size(-2)
        cos = self.cos[None, None, :x_BTHD.size(-2), :]
        sin = self.sin[None, None, :x_BTHD.size(-2), :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_BTHD)


# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for Grouped Query Attention"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention with optional sliding window"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.dropout = config.dropout
        self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # Position encoding (will be set by model)
        self.pos_encoding = None

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if self.pos_encoding is not None and isinstance(self.pos_encoding, (RoPE, YARN)):
            q = self.pos_encoding(q)
            k = self.pos_encoding(k)

        # Create attention mask for sliding window if specified
        attn_mask = None
        if self.sliding_window is not None and self.sliding_window > 0:
            attn_mask = _create_sliding_window_mask(seq_len, self.sliding_window, x.device)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None), dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


def _create_sliding_window_mask(seq_len: int, sliding_window: int, device) -> torch.Tensor:
    """Create a causal sliding window attention mask.

    For scaled_dot_product_attention with bool mask:
        True = ALLOWED to attend
        False = masked out (cannot attend)

    Returns:
        mask: (1, 1, seq_len, seq_len) bool tensor
    """
    # Start with causal mask: True on lower triangle (including diagonal)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    # Add sliding window constraint: mask out positions too far back
    for i in range(seq_len):
        if i > sliding_window:
            mask[i, :i-sliding_window] = False
    return mask.unsqueeze(0).unsqueeze(0)


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (single K/V head) with optional sliding window"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.dropout = config.dropout
        self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self.pos_encoding = None

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)

        if self.pos_encoding is not None and isinstance(self.pos_encoding, (RoPE, YARN)):
            q = self.pos_encoding(q)
            k = self.pos_encoding(k)

        k = repeat_kv(k, self.n_heads)
        v = repeat_kv(v, self.n_heads)

        # Create attention mask for sliding window if specified
        attn_mask = None
        if self.sliding_window is not None and self.sliding_window > 0:
            attn_mask = _create_sliding_window_mask(seq_len, self.sliding_window, x.device)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None), dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with optional sliding window"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k
        self.dropout = config.dropout
        self.sliding_window = config.sliding_window

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=config.attention_bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # QK normalization
        if config.norm_type == "rmsnorm":
            self.q_norm = nn.RMSNorm(self.d_k, eps=config.norm_eps)
            self.k_norm = nn.RMSNorm(self.d_k, eps=config.norm_eps)
        else:
            self.q_norm = nn.LayerNorm(self.d_k, eps=config.norm_eps)
            self.k_norm = nn.LayerNorm(self.d_k, eps=config.norm_eps)

        self.pos_encoding = None

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (batch, n_heads, seq_len, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.pos_encoding is not None and isinstance(self.pos_encoding, (RoPE, YARN)):
            q = self.pos_encoding(q)
            k = self.pos_encoding(k)

        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

        # Create attention mask for sliding window if specified
        attn_mask = None
        if self.sliding_window is not None and self.sliding_window > 0:
            attn_mask = _create_sliding_window_mask(seq_len, self.sliding_window, x.device)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None),  # Use is_causal only if no custom mask
            dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA)

    Compresses KV representations through a low-rank latent bottleneck
    to reduce KV cache size while maintaining quality. Used in DeepSeek-V2/V3.

    Architecture:
    - Q: Standard projection (d_model -> n_heads * d_k)
    - K/V: Compressed path
      1. Down-project: d_model -> d_latent (compression)
      2. Up-project: d_latent -> n_heads * d_k (expansion)

    Benefits:
    - Reduced KV cache: ~4x smaller than MHA
    - Parameter efficient: Similar to GQA
    - Quality: Can match MHA with proper d_latent tuning

    Reference: DeepSeek-V2 (https://arxiv.org/abs/2405.04434)
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_latent = config.d_latent
        self.dropout = config.dropout
        self.positional_encoding = config.positional_encoding
        self.sliding_window = config.sliding_window

        # Standard Q projection
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=config.attention_bias)

        # Latent compression for KV
        # Shared down-projection for both K and V
        self.kv_down = nn.Linear(self.d_model, self.d_latent, bias=False)

        # K and V up-projections depend on whether we're using RoPE/YARN
        # For RoPE/YARN: split K into RoPE and non-RoPE components (DeepSeek-V2 style)
        # For other encodings: standard projection
        if self.positional_encoding in ["rope", "yarn"]:
            self.d_rope_latent = config.d_rope_latent
            # Split K into RoPE and non-RoPE components
            self.k_rope_proj = nn.Linear(self.d_latent, self.n_heads * self.d_rope_latent, bias=False)
            self.k_nope_proj = nn.Linear(self.d_latent, self.n_heads * (self.d_k - self.d_rope_latent), bias=False)
        else:
            # Standard K projection for sinusoidal/alibi
            self.d_rope_latent = None
            self.k_proj = nn.Linear(self.d_latent, self.n_heads * self.d_k, bias=False)

        # V projection (always standard)
        self.v_proj = nn.Linear(self.d_latent, self.n_heads * self.d_k, bias=False)

        # Output projection
        self.w_o = nn.Linear(self.n_heads * self.d_k, self.d_model, bias=False)

        # QK normalization (important for stability)
        if config.norm_type == "rmsnorm":
            self.q_norm = nn.RMSNorm(self.d_k, eps=config.norm_eps)
            self.k_norm = nn.RMSNorm(self.d_k, eps=config.norm_eps)
        else:
            self.q_norm = nn.LayerNorm(self.d_k, eps=config.norm_eps)
            self.k_norm = nn.LayerNorm(self.d_k, eps=config.norm_eps)

        self.pos_encoding = None

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Q: Standard projection
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # K/V: Compress through latent bottleneck
        latent = self.kv_down(x)  # (batch, seq, d_latent)

        # V: Standard up-projection (always the same)
        v = self.v_proj(latent).view(batch_size, seq_len, self.n_heads, self.d_k)

        # K projection depends on positional encoding type
        if self.positional_encoding in ["rope", "yarn"]:
            # Split K into RoPE and non-RoPE components (DeepSeek-V2 style)
            k_rope = self.k_rope_proj(latent).view(batch_size, seq_len, self.n_heads, self.d_rope_latent)
            k_nope = self.k_nope_proj(latent).view(batch_size, seq_len, self.n_heads, self.d_k - self.d_rope_latent)

            # Apply QK normalization before RoPE
            q = self.q_norm(q)

            # Transpose to (batch, n_heads, seq_len, d_k)
            q = q.transpose(1, 2)
            k_rope = k_rope.transpose(1, 2)
            k_nope = k_nope.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply RoPE to Q and K_rope components
            if self.pos_encoding is not None and isinstance(self.pos_encoding, (RoPE, YARN)):
                # Split Q into RoPE and non-RoPE parts
                q_rope = q[..., :self.d_rope_latent]
                q_nope = q[..., self.d_rope_latent:]

                # Apply RoPE
                q_rope = self.pos_encoding(q_rope)
                k_rope = self.pos_encoding(k_rope)

                # Recombine Q
                q = torch.cat([q_rope, q_nope], dim=-1)

            # Concatenate K components
            k = torch.cat([k_rope, k_nope], dim=-1)

        else:
            # Standard path for sinusoidal/alibi (no RoPE splitting)
            k = self.k_proj(latent).view(batch_size, seq_len, self.n_heads, self.d_k)

            # Apply QK normalization
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Transpose to (batch, n_heads, seq_len, d_k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Note: Sinusoidal is added to embeddings, ALiBi is handled via attention bias
            # Neither requires special handling in attention mechanism

        # Apply K normalization (for RoPE path, needs to be after concat)
        if self.positional_encoding in ["rope", "yarn"]:
            k = k.transpose(1, 2)  # Back to (batch, seq_len, n_heads, d_k)
            k = self.k_norm(k)
            k = k.transpose(1, 2)  # Back to (batch, n_heads, seq_len, d_k)

        # Create attention mask for sliding window if specified
        attn_mask = None
        if self.sliding_window is not None and self.sliding_window > 0:
            attn_mask = _create_sliding_window_mask(seq_len, self.sliding_window, x.device)

        # Scaled dot-product attention (uses Flash Attention via PyTorch)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None),
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k)
        return self.w_o(attn_output)


# ============================================================================
# FEED-FORWARD ACTIVATIONS
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation (Swish-Gated Linear Unit)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GeGLU(nn.Module):
    """GeGLU activation (GELU-Gated Linear Unit)

    Used in T5, PaLM, and many modern LLMs. Often performs better than SwiGLU
    for certain tasks. Same structure as SwiGLU but with GELU activation.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.gelu(self.gate_proj(x)) * self.up_proj(x)))


class ReGLU(nn.Module):
    """ReGLU activation (ReLU-Gated Linear Unit)

    Simpler and faster than SwiGLU/GeGLU. Good baseline for comparison.
    Uses ReLU activation instead of smooth functions.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.relu(self.gate_proj(x)) * self.up_proj(x)))


class StandardFFN(nn.Module):
    """Standard Feed-Forward Network with configurable activation"""
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "leaky_relu":
            self.activation = lambda x: F.leaky_relu(x, 0.01)
        else:
            self.activation = F.gelu

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MoEFFN(nn.Module):
    """
    Mixture of Experts Feed-Forward Network

    Replaces standard FFN with multiple expert FFNs and a learned router.
    Each token is routed to top-K experts based on router scores.

    Architecture:
    - Router: Linear layer that computes expert scores for each token
    - Experts: num_experts copies of the chosen FFN type (SwiGLU, GeGLU, etc.)
    - Top-K Gating: Each token is processed by num_experts_per_token experts
    - Load Balancing: Auxiliary loss to encourage even expert utilization

    Used in: Mixtral, DeepSeek, Switch Transformer, GShard

    Args:
        config: ModelConfig with MoE parameters
            - num_experts: Number of expert FFNs
            - num_experts_per_token: Top-K routing (usually 2)
            - activation: Type of FFN to use for each expert
            - d_ff: Hidden dimension for each expert
            - load_balancing_loss_weight: Weight for load balancing loss
            - router_z_loss_weight: Weight for router z-loss
    """
    def __init__(self, config):
        super().__init__()
        from config import ModelConfig

        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.load_balancing_loss_weight = config.load_balancing_loss_weight
        self.router_z_loss_weight = config.router_z_loss_weight

        # Router: learns which experts to use for each token
        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)

        # Experts: create num_experts copies of the same FFN type
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            if config.activation == 'swiglu':
                expert = SwiGLU(config.d_model, config.d_ff, config.dropout)
            elif config.activation == 'geglu':
                expert = GeGLU(config.d_model, config.d_ff, config.dropout)
            elif config.activation == 'reglu':
                expert = ReGLU(config.d_model, config.d_ff, config.dropout)
            else:
                expert = StandardFFN(config.d_model, config.d_ff, config.activation, config.dropout)
            self.experts.append(expert)

    def forward(self, x):
        """
        Forward pass with top-K expert routing

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: Scalar auxiliary loss (load balancing + z-loss)
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)

        # Router computes expert scores
        router_logits = self.router(x_flat)  # (batch * seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (batch * seq_len, num_experts)

        # Top-K selection: pick top num_experts_per_token experts for each token
        topk_probs, topk_indices = torch.topk(router_probs, k=self.num_experts_per_token, dim=-1)
        # topk_probs: (batch * seq_len, num_experts_per_token)
        # topk_indices: (batch * seq_len, num_experts_per_token)

        # Normalize topk probabilities (so they sum to 1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        # Strategy: For each token, compute outputs from selected experts and combine
        output = torch.zeros_like(x_flat)  # (batch * seq_len, d_model)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens route to this expert
            expert_mask = (topk_indices == expert_idx)  # (batch * seq_len, num_experts_per_token)
            token_expert_mask = expert_mask.any(dim=-1)  # (batch * seq_len,) - tokens using this expert

            if not token_expert_mask.any():
                continue  # No tokens route to this expert

            # Get tokens for this expert
            expert_tokens = x_flat[token_expert_mask]  # (num_tokens_for_expert, d_model)

            # Compute expert output
            expert_out = self.experts[expert_idx](expert_tokens)  # (num_tokens_for_expert, d_model)

            # Get routing weights for this expert
            # Find position of this expert in topk for each token
            expert_weights = torch.zeros(token_expert_mask.sum(), device=x.device)
            for k in range(self.num_experts_per_token):
                mask_k = expert_mask[token_expert_mask, k]  # Tokens where this expert is in position k
                expert_weights[mask_k] = topk_probs[token_expert_mask, k][mask_k]

            # Add weighted expert output to final output
            output[token_expert_mask] += expert_weights.unsqueeze(-1) * expert_out

        # Reshape back to (batch, seq_len, d_model)
        output = output.view(batch_size, seq_len, d_model)

        # Compute auxiliary losses
        aux_loss = self._compute_aux_loss(router_logits, router_probs, topk_indices)

        return output, aux_loss

    def _compute_aux_loss(self, router_logits, router_probs, topk_indices):
        """
        Compute auxiliary losses for MoE training

        1. Load Balancing Loss: Encourages even distribution of tokens across experts
        2. Router Z-Loss: Prevents overconfident routing (large logits)

        Args:
            router_logits: (batch * seq_len, num_experts) - raw router outputs
            router_probs: (batch * seq_len, num_experts) - softmax probabilities
            topk_indices: (batch * seq_len, num_experts_per_token) - selected expert indices

        Returns:
            aux_loss: Scalar tensor
        """
        num_tokens = router_probs.shape[0]

        # 1. Load Balancing Loss (Switch Transformer formulation)
        # Encourages: P(expert) * fraction_tokens_to_expert to be uniform

        # f_i: Fraction of tokens assigned to expert i (based on top-k selection)
        expert_counts = torch.zeros(self.num_experts, device=router_probs.device)
        for i in range(self.num_experts):
            expert_counts[i] = (topk_indices == i).sum()
        f_i = expert_counts / (num_tokens * self.num_experts_per_token)  # Normalize

        # P_i: Average router probability for expert i (over all tokens)
        P_i = router_probs.mean(dim=0)  # (num_experts,)

        # Load balancing loss: num_experts * sum(f_i * P_i)
        # This is minimized when f_i and P_i are uniform (both = 1/num_experts)
        load_balance_loss = self.num_experts * (f_i * P_i).sum()

        # 2. Router Z-Loss (optional, for stability)
        # Penalizes large router logits to prevent overconfidence
        # z_loss = mean(log(sum(exp(router_logits)))^2)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        # Combine losses with weights
        aux_loss = (
            self.load_balancing_loss_weight * load_balance_loss +
            self.router_z_loss_weight * router_z_loss
        )

        return aux_loss


# ============================================================================
# MAMBA2 SSM BLOCK
# ============================================================================

class Mamba2(nn.Module):
    """
    Custom Mamba2 State Space Model implementation in pure PyTorch

    This is a self-contained implementation that doesn't require the mamba-ssm package.
    It uses standard PyTorch operations and runs on GPU efficiently.

    Architecture:
    - Input projection with expansion
    - 1D causal convolution for local context
    - Selective SSM (State Space Model) with learned dynamics
    - Gated activation (SiLU)
    - Output projection

    Reference: https://arxiv.org/abs/2405.21060
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        dt_rank: int = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.ngroups = ngroups

        # Compute dt_rank (rank of delta projection)
        if dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        # Input projection: d_model -> d_inner * 2 (for gating)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Convolutional layer for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise convolution
            padding=d_conv - 1,  # Causal padding
            bias=True
        )

        # SSM parameters
        # A: State transition matrix (fixed, initialized with special properties)
        A = torch.randn(self.d_inner, self.d_state)
        self.A_log = nn.Parameter(torch.log(A))  # Log-space for numerical stability

        # D: Skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Time-step projection (delta)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj with special initialization for stability
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias to produce values between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # X projection to dt_rank
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank, bias=False)

        # B and C projections (input-dependent SSM parameters)
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection with gating
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_input, z = xz.chunk(2, dim=-1)  # Each is (batch, seq_len, d_inner)

        # 1D Convolution (causal)
        x_conv = self.conv1d(x_input.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)  # Activation after conv

        # Selective SSM
        # Compute time-step delta
        x_dt = self.x_proj(x_conv)  # (batch, seq_len, dt_rank)
        dt = self.dt_proj(x_dt)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)  # Ensure positive time-steps

        # Compute input-dependent B and C
        B = self.B_proj(x_conv)  # (batch, seq_len, d_state)
        C = self.C_proj(x_conv)  # (batch, seq_len, d_state)

        # Get A from log-space
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan (sequential implementation)
        y = self._selective_scan(x_conv, dt, A, B, C)

        # Skip connection (like residual)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gating with z
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def _selective_scan(self, x, dt, A, B, C):
        """
        Selective scan implementation (sequential for correctness)

        WARNING: This is a memory-intensive fallback implementation!
        For production use, install mamba-ssm for optimized CUDA kernels:
        pip install mamba-ssm>=2.0.0 causal-conv1d>=1.2.0

        Args:
            x: (batch, seq_len, d_inner) - input sequence
            dt: (batch, seq_len, d_inner) - time-step deltas
            A: (d_inner, d_state) - state transition matrix
            B: (batch, seq_len, d_state) - input matrix
            C: (batch, seq_len, d_state) - output matrix

        Returns:
            y: (batch, seq_len, d_inner) - output sequence
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)

        # Output buffer
        y = []

        # Sequential scan (not optimized but correct)
        # Process in chunks to reduce memory (less gradient tracking)
        A_unsqueezed = A.unsqueeze(0)  # Pre-compute (1, d_inner, d_state)

        for t in range(seq_len):
            # Discretize A using dt: A_discrete = exp(dt * A)
            # For numerical stability, use A_discrete ≈ (1 + dt * A)
            dt_t = dt[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)

            # Discretized state transition: h = A_discrete * h + B * x
            # Simplified: h = h + dt * (A * h + B * x)
            B_t = B[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            x_t = x[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)

            # State update: h_new = h * exp(dt * A) + B * x * dt
            # Approximation: h_new ≈ h + dt * A * h + dt * B * x
            dh_state = dt_t * (A_unsqueezed * h)  # (batch, d_inner, d_state)
            dh_input = dt_t * (B_t * x_t)  # (batch, d_inner, d_state)
            h = h + dh_state + dh_input

            # Output: y = C * h
            C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            y_t = (C_t * h).sum(dim=-1)  # (batch, d_inner)

            # Detach and reattach to reduce gradient graph depth
            # This trades off gradient accuracy for memory
            if t > 0 and t % 64 == 0 and self.training:
                # Checkpoint state every 64 steps to limit gradient graph
                h = h.detach()
                h.requires_grad_(True)

            y.append(y_t)

        return torch.stack(y, dim=1)


class Mamba2Block(nn.Module):
    """
    Mamba2 State Space Model block with normalization and residual connection

    Uses official mamba-ssm library for optimized CUDA kernels.
    Much more memory efficient than custom PyTorch implementation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Normalization before Mamba2 (pre-norm architecture)
        NormClass = NORM_TYPES[config.norm_type]
        self.norm = NormClass(config.d_model, eps=config.norm_eps)

        # Try to use official mamba-ssm implementation (optimized CUDA kernels)
        # Fall back to custom PyTorch implementation if not available
        try:
            from mamba_ssm import Mamba2 as Mamba2Official
            self.mamba = Mamba2Official(
                d_model=config.d_model,
                d_state=config.state_size,
                d_conv=config.conv_kernel_size,
                expand=config.expand_factor,
                headdim=config.headdim,
                ngroups=config.ngroups,
                chunk_size=config.chunk_size,
            )
            print("Using official mamba-ssm optimized kernels")
        except ImportError:
            print("Warning: mamba-ssm not found. Using PyTorch fallback implementation.")
            print("For better performance, install: pip install mamba-ssm>=2.0.0 causal-conv1d>=1.2.0")
            self.mamba = Mamba2(
                d_model=config.d_model,
                d_state=config.state_size,
                d_conv=config.conv_kernel_size,
                expand=config.expand_factor,
                headdim=config.headdim,
                ngroups=config.ngroups,
                chunk_size=config.chunk_size,
            )

    def forward(self, x):
        """
        Forward pass with residual connection

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x: (batch, seq_len, d_model)
        """
        # Pre-norm with residual connection
        return x + self.mamba(self.norm(x))


# ============================================================================
# REGISTRIES
# ============================================================================

POSITIONAL_ENCODINGS = {
    'learned': LearnedPositional,
    'sinusoidal': SinusoidalPositional,
    'rope': RoPE,
    'alibi': ALiBi,
    'yarn': YARN
}

ATTENTION_TYPES = {
    'mha': MultiHeadAttention,
    'mqa': MultiQueryAttention,
    'gqa': GroupedQueryAttention,
    'mla': MultiHeadLatentAttention
}

NORM_TYPES = {
    'layernorm': nn.LayerNorm,
    'rmsnorm': nn.RMSNorm
}

ACTIVATION_TYPES = {
    'relu': 'relu',
    'gelu': 'gelu',
    'silu': 'silu',
    'leaky_relu': 'leaky_relu',
    'swiglu': SwiGLU,
    'geglu': GeGLU,
    'reglu': ReGLU
}
