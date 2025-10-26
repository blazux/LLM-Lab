import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# POSITIONAL ENCODINGS
# ============================================================================

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
    """Standard Multi-Head Attention"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.dropout = config.dropout

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

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (single K/V head)"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.dropout = config.dropout

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

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups
        self.d_k = config.d_k
        self.dropout = config.dropout

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

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
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


# ============================================================================
# MAMBA2 SSM BLOCK
# ============================================================================

class Mamba2Block(nn.Module):
    """
    Mamba2 State Space Model block wrapper

    This wraps the optimized mamba-ssm implementation from:
    https://github.com/state-spaces/mamba

    Note: Requires mamba-ssm package:
        pip install mamba-ssm causal-conv1d
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            from mamba_ssm import Mamba2
        except ImportError:
            raise ImportError(
                "mamba-ssm package is required for Mamba2 models.\n"
                "Install with: pip install mamba-ssm causal-conv1d>=1.2.0\n"
                "See: https://github.com/state-spaces/mamba"
            )

        # Normalization before Mamba2 (pre-norm architecture)
        NormClass = NORM_TYPES[config.norm_type]
        self.norm = NormClass(config.d_model, eps=config.norm_eps)

        # Mamba2 SSM layer with optimized CUDA kernels
        self.mamba = Mamba2(
            d_model=config.d_model,
            d_state=config.state_size,
            d_conv=config.conv_kernel_size,
            expand=config.expand_factor,
            headdim=64,  # Standard head dimension for Mamba2
            ngroups=1,  # Number of groups for multi-head SSM (1 = single head)
            chunk_size=256,  # Chunk size for parallel scan
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
    'sinusoidal': SinusoidalPositional,
    'rope': RoPE,
    'alibi': ALiBi,
    'yarn': YARN
}

ATTENTION_TYPES = {
    'mha': MultiHeadAttention,
    'mqa': MultiQueryAttention,
    'gqa': GroupedQueryAttention
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
    'swiglu': SwiGLU
}
