import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from flash_attn import flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


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

    def forward(self, x, start_pos: int = 0):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(0)
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

    def forward(self, x, start_pos: int = 0):
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        return x + self.pe[start_pos:start_pos + seq_len, :]


class RoPE(nn.Module):
    """Rotary Position Embedding with configurable partial rotation.

    fraction = share of rotation pairs that actually rotate (the rest are
    position-free, GPT-NeoX / Qwen3-Next "partial RoPE" style). The default
    0.5 reproduces the historical quarter-spectrum behavior of this codebase.
    """
    def __init__(self, d_k: int, max_seq_len: int, fraction: float = 0.5):
        super().__init__()
        n_pairs = d_k // 2
        n_active = min(n_pairs, max(1, round(n_pairs * fraction)))
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=n_active, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(n_pairs - n_active)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor, start_pos: int = 0):
        """x: (batch, n_heads, seq_len, d_k)"""
        seq_len = x_BTHD.size(-2)
        assert self.cos.size(0) >= start_pos + seq_len
        cos = self.cos[None, None, start_pos:start_pos + seq_len, :]
        sin = self.sin[None, None, start_pos:start_pos + seq_len, :]
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
        """Returns bias for attention: (n_heads, seq_len, seq_len) with causal mask included.

        ALiBi bias for query i attending to key j is: -slope * (i - j).
        Future positions (j > i) are set to -inf (causal mask).
        """
        query_idx = torch.arange(seq_len, device=self.bias.device).unsqueeze(1)  # (seq_len, 1)
        key_idx = torch.arange(seq_len, device=self.bias.device).unsqueeze(0)    # (1, seq_len)
        distance = (query_idx - key_idx).clamp(min=0)  # (seq_len, seq_len)
        rel_bias = self.bias[:, distance]  # (n_heads, seq_len, seq_len)
        causal_mask = query_idx < key_idx  # True where j > i (future)
        rel_bias = rel_bias.masked_fill(causal_mask, float('-inf'))
        return rel_bias  # (n_heads, seq_len, seq_len)

    def get_slopes(self) -> torch.Tensor:
        """Return positive slope values (n_heads,) for flash_attn alibi_slopes parameter."""
        return -self.bias[:, 1].float()  # flash_attn requires fp32 slopes


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

    def forward(self, x_BTHD: torch.Tensor, start_pos: int = 0):
        """x: (batch, n_heads, seq_len, d_k)"""
        seq_len = x_BTHD.size(-2)
        assert self.cos.size(0) >= start_pos + seq_len
        cos = self.cos[None, None, start_pos:start_pos + seq_len, :]
        sin = self.sin[None, None, start_pos:start_pos + seq_len, :]
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

        # Qwen3-Next-style sigmoid output gate (query-dependent, elementwise)
        if getattr(config, 'use_attention_gate', False):
            self.attn_gate = nn.Linear(config.d_model, self.n_heads * self.d_k, bias=False)
        else:
            self.attn_gate = None

    def forward(self, x, past_key_values=None, start_pos=0, doc_ids=None):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if self.pos_encoding is not None and isinstance(self.pos_encoding, (RoPE, YARN)):
            q = self.pos_encoding(q, start_pos)
            k = self.pos_encoding(k, start_pos)

        # KV cache: append new K/V to past; truncate only when decoding (cache active)
        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=2)
            v = torch.cat([past_key_values[1], v], dim=2)
            if self.sliding_window is not None and self.sliding_window > 0 and k.size(2) > self.sliding_window:
                k = k[:, :, -(self.sliding_window + 1):, :]
                v = v[:, :, -(self.sliding_window + 1):, :]
        new_kv = (k.detach(), v.detach())
        kv_len = k.size(2)

        # Flash attention for full-sequence sliding window passes (no KV cache active)
        use_flash = (
            _FLASH_ATTN_AVAILABLE
            and self.sliding_window is not None and self.sliding_window > 0
            and past_key_values is None
            and doc_ids is None
            and x.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash:
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            alibi_slopes = self.pos_encoding.get_slopes() if isinstance(self.pos_encoding, ALiBi) else None
            fa_kwargs = dict(dropout_p=self.dropout if self.training else 0.0, causal=True, window_size=(self.sliding_window, 0))
            if alibi_slopes is not None:
                fa_kwargs["alibi_slopes"] = alibi_slopes
            attn_output = flash_attn_func(q_fa, k_fa, v_fa, **fa_kwargs)
            attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
            if self.attn_gate is not None:
                attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
            return self.w_o(attn_output), new_kv

        # SDPA fallback path
        attn_mask = None
        is_causal = past_key_values is None  # during decode, all K/V are past so no masking needed

        if past_key_values is None and self.sliding_window is not None and self.sliding_window > 0:
            attn_mask = _create_sliding_window_mask(seq_len, self.sliding_window, x.device)
            is_causal = False

        if doc_ids is not None and past_key_values is None:
            doc_mask = _create_document_mask(doc_ids)
            attn_mask = doc_mask if attn_mask is None else (attn_mask & doc_mask)
            is_causal = False

        # Chunked prefill with a cache: without a mask, tokens inside the new
        # chunk could attend to LATER tokens of the same chunk
        if past_key_values is not None and seq_len > 1:
            past_len = kv_len - seq_len
            q_pos = torch.arange(seq_len, device=x.device).unsqueeze(1) + past_len
            k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
            chunk_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
            attn_mask = chunk_mask if attn_mask is None else (attn_mask & chunk_mask)
            is_causal = False

        if self.pos_encoding is not None and isinstance(self.pos_encoding, ALiBi):
            if past_key_values is not None:
                # Decode step: q at position (kv_len-1) attending to all cached keys
                key_idx = torch.arange(kv_len, device=x.device)
                distances = (kv_len - 1 - key_idx).clamp(min=0)
                attn_bias = self.pos_encoding.bias[:, distances].unsqueeze(0).unsqueeze(2)  # (1, n_heads, 1, kv_len)
            else:
                attn_bias = self.pos_encoding.get_bias(kv_len).unsqueeze(0)
                if attn_mask is not None:
                    attn_bias = attn_bias.masked_fill(~attn_mask, float('-inf'))
            attn_mask = attn_bias
            is_causal = False

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        if self.attn_gate is not None:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
        return self.w_o(attn_output), new_kv


def _create_document_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Create a causal, intra-document attention mask for packed sequences.

    Tokens may only attend to earlier tokens of the SAME document — without
    this, packing multiple documents into one sequence lets attention leak
    across document boundaries (Llama 3 / SmolLM3 style masking).

    Args:
        doc_ids: (batch, seq_len) int tensor of per-token document indices

    Returns:
        mask: (batch, 1, seq_len, seq_len) bool tensor (True = allowed)
    """
    seq_len = doc_ids.size(1)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=doc_ids.device))
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, T, T)
    return (causal.unsqueeze(0) & same_doc).unsqueeze(1)


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

        # Qwen3-Next-style sigmoid output gate (query-dependent, elementwise)
        if getattr(config, 'use_attention_gate', False):
            self.attn_gate = nn.Linear(config.d_model, self.n_heads * self.d_k, bias=False)
        else:
            self.attn_gate = None

    def forward(self, x, past_key_values=None, start_pos=0, doc_ids=None):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)

        if self.pos_encoding is not None and isinstance(self.pos_encoding, (RoPE, YARN)):
            q = self.pos_encoding(q, start_pos)
            k = self.pos_encoding(k, start_pos)

        # KV cache (compact single-head k/v, before repeat); truncate only when decoding
        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=2)
            v = torch.cat([past_key_values[1], v], dim=2)
            if self.sliding_window is not None and self.sliding_window > 0 and k.size(2) > self.sliding_window:
                k = k[:, :, -(self.sliding_window + 1):, :]
                v = v[:, :, -(self.sliding_window + 1):, :]
        new_kv = (k.detach(), v.detach())
        kv_len = k.size(2)

        # Flash attention (MQA natively supported, no repeat needed)
        use_flash = (
            _FLASH_ATTN_AVAILABLE
            and self.sliding_window is not None and self.sliding_window > 0
            and past_key_values is None
            and doc_ids is None
            and x.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash:
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            alibi_slopes = self.pos_encoding.get_slopes() if isinstance(self.pos_encoding, ALiBi) else None
            fa_kwargs = dict(dropout_p=self.dropout if self.training else 0.0, causal=True, window_size=(self.sliding_window, 0))
            if alibi_slopes is not None:
                fa_kwargs["alibi_slopes"] = alibi_slopes
            attn_output = flash_attn_func(q_fa, k_fa, v_fa, **fa_kwargs)
            attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
            if self.attn_gate is not None:
                attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
            return self.w_o(attn_output), new_kv

        # SDPA fallback: expand k/v to full heads
        k_expanded = repeat_kv(k, self.n_heads)
        v_expanded = repeat_kv(v, self.n_heads)

        attn_mask = None
        is_causal = past_key_values is None

        if past_key_values is None and self.sliding_window is not None and self.sliding_window > 0:
            attn_mask = _create_sliding_window_mask(seq_len, self.sliding_window, x.device)
            is_causal = False

        if doc_ids is not None and past_key_values is None:
            doc_mask = _create_document_mask(doc_ids)
            attn_mask = doc_mask if attn_mask is None else (attn_mask & doc_mask)
            is_causal = False

        # Chunked prefill with a cache: without a mask, tokens inside the new
        # chunk could attend to LATER tokens of the same chunk
        if past_key_values is not None and seq_len > 1:
            past_len = kv_len - seq_len
            q_pos = torch.arange(seq_len, device=x.device).unsqueeze(1) + past_len
            k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
            chunk_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
            attn_mask = chunk_mask if attn_mask is None else (attn_mask & chunk_mask)
            is_causal = False

        if self.pos_encoding is not None and isinstance(self.pos_encoding, ALiBi):
            if past_key_values is not None:
                key_idx = torch.arange(kv_len, device=x.device)
                distances = (kv_len - 1 - key_idx).clamp(min=0)
                attn_bias = self.pos_encoding.bias[:, distances].unsqueeze(0).unsqueeze(2)
            else:
                attn_bias = self.pos_encoding.get_bias(kv_len).unsqueeze(0)
                if attn_mask is not None:
                    attn_bias = attn_bias.masked_fill(~attn_mask, float('-inf'))
            attn_mask = attn_bias
            is_causal = False

        attn_output = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded, attn_mask=attn_mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        if self.attn_gate is not None:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
        return self.w_o(attn_output), new_kv


class LightningIndexer(nn.Module):
    """DSA-style lightning indexer (DeepSeek V3.2 / GLM-5).

    A tiny scoring network deciding which past tokens each query attends to:
        I(i, j) = sum_h w_h(i) * ReLU(q_idx_h(i) . k_idx(j))
    Main attention then runs only over each query's top-k tokens. The indexer
    is trained with a KL alignment loss against the (detached) true attention
    distribution, returned as an auxiliary loss.
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_index: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.d_index = d_index
        self.q_proj = nn.Linear(d_model, n_heads * d_index, bias=False)
        self.k_proj = nn.Linear(d_model, d_index, bias=False)
        self.w_proj = nn.Linear(d_model, n_heads, bias=False)

    def scores(self, x):
        """x: (B, T, d_model) -> index scores (B, T, T) (query, key)"""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_index)
        k = self.k_proj(x)  # (B, T, d_index)
        w = self.w_proj(x)  # (B, T, n_heads)
        # (B, H, T, T): relu(q . k) per indexer head, then weighted head sum
        qk = torch.einsum('bthd,bsd->bhts', q, k)
        return torch.einsum('bth,bhts->bts', w, F.relu(qk))


def _topk_sparse_mask(index_scores: torch.Tensor, top_k: int,
                      constraint: torch.Tensor) -> torch.Tensor:
    """Bool mask (B, 1, T, T): key j is in query i's top-k among ALLOWED keys.

    constraint: (B|1, 1, T, T) bool — causal ∧ sliding-window ∧ document mask.
    Selecting from the constrained scores (not just causal) prevents the fatal
    case where top-k picks only tokens another mask forbids, leaving a query
    with zero attendable keys (SDPA: zero rows on CPU, NaN on CUDA). The
    diagonal is always included so every query can at least attend to itself.
    """
    B, T, _ = index_scores.shape
    c = constraint.squeeze(1)  # (B|1, T, T)
    scores = index_scores.masked_fill(~c, float('-inf'))
    k = min(top_k, T)
    topk_idx = scores.topk(k, dim=-1).indices
    mask = torch.zeros(B, T, T, dtype=torch.bool, device=index_scores.device)
    mask.scatter_(-1, topk_idx, True)
    mask = mask & c  # drop picks that were -inf placeholders in short rows
    mask.diagonal(dim1=-2, dim2=-1).fill_(True)
    return mask.unsqueeze(1)


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

        # Qwen3-Next-style sigmoid output gate (query-dependent, elementwise)
        if getattr(config, 'use_attention_gate', False):
            self.attn_gate = nn.Linear(config.d_model, self.n_heads * self.d_k, bias=False)
        else:
            self.attn_gate = None

        # DSA-style sparse attention (indexer + top-k token selection)
        if getattr(config, 'use_sparse_attention', False):
            self.indexer = LightningIndexer(
                config.d_model,
                n_heads=getattr(config, 'sparse_indexer_heads', 4),
                d_index=getattr(config, 'sparse_indexer_dim', 32),
            )
            self.sparse_top_k = getattr(config, 'sparse_top_k', 512)
        else:
            self.indexer = None
        self.indexer_loss = None

    def forward(self, x, past_key_values=None, start_pos=0, doc_ids=None):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        # QK normalization (on last dim, works in NHWD layout)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (batch, n_heads, seq_len, d_k) for RoPE and SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.pos_encoding is not None and isinstance(self.pos_encoding, (RoPE, YARN)):
            q = self.pos_encoding(q, start_pos)
            k = self.pos_encoding(k, start_pos)

        # KV cache (compact n_kv_heads, before repeat); truncate only when decoding
        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=2)
            v = torch.cat([past_key_values[1], v], dim=2)
            if self.sliding_window is not None and self.sliding_window > 0 and k.size(2) > self.sliding_window:
                k = k[:, :, -(self.sliding_window + 1):, :]
                v = v[:, :, -(self.sliding_window + 1):, :]
        new_kv = (k.detach(), v.detach())
        kv_len = k.size(2)

        # Flash attention (GQA natively supported, no repeat needed)
        use_flash = (
            _FLASH_ATTN_AVAILABLE
            and self.sliding_window is not None and self.sliding_window > 0
            and past_key_values is None
            and doc_ids is None
            and self.indexer is None
            and x.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash:
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            alibi_slopes = self.pos_encoding.get_slopes() if isinstance(self.pos_encoding, ALiBi) else None
            fa_kwargs = dict(dropout_p=self.dropout if self.training else 0.0, causal=True, window_size=(self.sliding_window, 0))
            if alibi_slopes is not None:
                fa_kwargs["alibi_slopes"] = alibi_slopes
            attn_output = flash_attn_func(q_fa, k_fa, v_fa, **fa_kwargs)
            attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
            if self.attn_gate is not None:
                attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
            return self.w_o(attn_output), new_kv

        # SDPA fallback: expand k/v to full heads
        k_expanded = repeat_kv(k, self.n_kv_groups)
        v_expanded = repeat_kv(v, self.n_kv_groups)

        attn_mask = None
        is_causal = past_key_values is None

        if past_key_values is None and self.sliding_window is not None and self.sliding_window > 0:
            attn_mask = _create_sliding_window_mask(seq_len, self.sliding_window, x.device)
            is_causal = False

        if doc_ids is not None and past_key_values is None:
            doc_mask = _create_document_mask(doc_ids)
            attn_mask = doc_mask if attn_mask is None else (attn_mask & doc_mask)
            is_causal = False

        # Chunked prefill with a cache: without a mask, tokens inside the new
        # chunk could attend to LATER tokens of the same chunk
        if past_key_values is not None and seq_len > 1:
            past_len = kv_len - seq_len
            q_pos = torch.arange(seq_len, device=x.device).unsqueeze(1) + past_len
            k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
            chunk_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
            attn_mask = chunk_mask if attn_mask is None else (attn_mask & chunk_mask)
            is_causal = False

        # DSA-style sparse attention: restrict each query to its indexer top-k
        # (decode with cache stays dense — a documented train/infer asymmetry:
        # with T < top_k they are identical anyway)
        self.indexer_loss = None
        if self.indexer is not None and past_key_values is None:
            index_scores = self.indexer.scores(x)
            causal4 = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool,
                                            device=x.device)).unsqueeze(0).unsqueeze(0)
            constraint = causal4 if attn_mask is None else (attn_mask & causal4)
            attn_mask = _topk_sparse_mask(index_scores, self.sparse_top_k, constraint)
            is_causal = False

            if self.training:
                # Train the indexer: KL(true attention dist || indexer dist),
                # true distribution detached (indexer follows the model) and
                # computed under the SAME constraints the real attention sees
                c = constraint.squeeze(1)  # (B|1, T, T)
                with torch.no_grad():
                    true_scores = (q @ k_expanded.transpose(-2, -1)) * (self.d_k ** -0.5)
                    true_scores = true_scores.masked_fill(~constraint, float('-inf'))
                    true_probs = F.softmax(true_scores.float(), dim=-1).mean(dim=1)  # (B, T, T)
                idx_logp = F.log_softmax(
                    index_scores.float().masked_fill(~c, float('-inf')), dim=-1)
                # Manual KL over valid entries only: masked positions have
                # true_prob 0 and log-prob -inf, and 0 * -inf would be NaN
                idx_logp = idx_logp.masked_fill(~c, 0.0)
                true_logp = torch.log(true_probs.clamp_min(1e-9)).masked_fill(~c, 0.0)
                self.indexer_loss = (true_probs * (true_logp - idx_logp)).sum(-1).mean()

        if self.pos_encoding is not None and isinstance(self.pos_encoding, ALiBi):
            if past_key_values is not None:
                key_idx = torch.arange(kv_len, device=x.device)
                distances = (kv_len - 1 - key_idx).clamp(min=0)
                attn_bias = self.pos_encoding.bias[:, distances].unsqueeze(0).unsqueeze(2)
            else:
                attn_bias = self.pos_encoding.get_bias(kv_len).unsqueeze(0)
                if attn_mask is not None:
                    attn_bias = attn_bias.masked_fill(~attn_mask, float('-inf'))
            attn_mask = attn_bias
            is_causal = False

        attn_output = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        if self.attn_gate is not None:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
        return self.w_o(attn_output), new_kv


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

        # Qwen3-Next-style sigmoid output gate (query-dependent, elementwise)
        if getattr(config, 'use_attention_gate', False):
            self.attn_gate = nn.Linear(config.d_model, self.n_heads * self.d_k, bias=False)
        else:
            self.attn_gate = None

    def forward(self, x, past_key_values=None, start_pos=0, doc_ids=None):
        batch_size, seq_len = x.size(0), x.size(1)

        # Q: Standard projection
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # K/V: Compress through latent bottleneck
        latent = self.kv_down(x)

        # V: Standard up-projection
        v = self.v_proj(latent).view(batch_size, seq_len, self.n_heads, self.d_k)

        # K projection depends on positional encoding type
        if self.positional_encoding in ["rope", "yarn"]:
            k_rope = self.k_rope_proj(latent).view(batch_size, seq_len, self.n_heads, self.d_rope_latent)
            k_nope = self.k_nope_proj(latent).view(batch_size, seq_len, self.n_heads, self.d_k - self.d_rope_latent)

            # QK norm on Q before transpose
            q = self.q_norm(q)

            # Transpose to (batch, n_heads, seq_len, d_k)
            q = q.transpose(1, 2)
            k_rope = k_rope.transpose(1, 2)
            k_nope = k_nope.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply RoPE with start_pos
            if self.pos_encoding is not None and isinstance(self.pos_encoding, (RoPE, YARN)):
                q_rope = q[..., :self.d_rope_latent]
                q_nope = q[..., self.d_rope_latent:]
                q_rope = self.pos_encoding(q_rope, start_pos)
                k_rope = self.pos_encoding(k_rope, start_pos)
                q = torch.cat([q_rope, q_nope], dim=-1)

            k = torch.cat([k_rope, k_nope], dim=-1)

            # K normalization (after concat, in NCHW layout)
            k = k.transpose(1, 2)
            k = self.k_norm(k)
            k = k.transpose(1, 2)

        else:
            # Standard path for sinusoidal/alibi
            k = self.k_proj(latent).view(batch_size, seq_len, self.n_heads, self.d_k)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # KV cache; truncate only when decoding (cache active)
        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=2)
            v = torch.cat([past_key_values[1], v], dim=2)
            if self.sliding_window is not None and self.sliding_window > 0 and k.size(2) > self.sliding_window:
                k = k[:, :, -(self.sliding_window + 1):, :]
                v = v[:, :, -(self.sliding_window + 1):, :]
        new_kv = (k.detach(), v.detach())
        kv_len = k.size(2)

        # Flash attention for full-sequence sliding window passes
        use_flash = (
            _FLASH_ATTN_AVAILABLE
            and self.sliding_window is not None and self.sliding_window > 0
            and past_key_values is None
            and doc_ids is None
            and x.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash:
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            alibi_slopes = self.pos_encoding.get_slopes() if isinstance(self.pos_encoding, ALiBi) else None
            fa_kwargs = dict(dropout_p=self.dropout if self.training else 0.0, causal=True, window_size=(self.sliding_window, 0))
            if alibi_slopes is not None:
                fa_kwargs["alibi_slopes"] = alibi_slopes
            attn_output = flash_attn_func(q_fa, k_fa, v_fa, **fa_kwargs)
            attn_output = attn_output.reshape(batch_size, seq_len, self.n_heads * self.d_k)
            if self.attn_gate is not None:
                attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
            return self.w_o(attn_output), new_kv

        # SDPA fallback
        attn_mask = None
        is_causal = past_key_values is None

        if past_key_values is None and self.sliding_window is not None and self.sliding_window > 0:
            attn_mask = _create_sliding_window_mask(seq_len, self.sliding_window, x.device)
            is_causal = False

        if doc_ids is not None and past_key_values is None:
            doc_mask = _create_document_mask(doc_ids)
            attn_mask = doc_mask if attn_mask is None else (attn_mask & doc_mask)
            is_causal = False

        # Chunked prefill with a cache: without a mask, tokens inside the new
        # chunk could attend to LATER tokens of the same chunk
        if past_key_values is not None and seq_len > 1:
            past_len = kv_len - seq_len
            q_pos = torch.arange(seq_len, device=x.device).unsqueeze(1) + past_len
            k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
            chunk_mask = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
            attn_mask = chunk_mask if attn_mask is None else (attn_mask & chunk_mask)
            is_causal = False

        if self.pos_encoding is not None and isinstance(self.pos_encoding, ALiBi):
            if past_key_values is not None:
                key_idx = torch.arange(kv_len, device=x.device)
                distances = (kv_len - 1 - key_idx).clamp(min=0)
                attn_bias = self.pos_encoding.bias[:, distances].unsqueeze(0).unsqueeze(2)
            else:
                attn_bias = self.pos_encoding.get_bias(kv_len).unsqueeze(0)
                if attn_mask is not None:
                    attn_bias = attn_bias.masked_fill(~attn_mask, float('-inf'))
            attn_mask = attn_bias
            is_causal = False

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_k)
        if self.attn_gate is not None:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
        return self.w_o(attn_output), new_kv


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
        elif activation == "relu2":
            # Squared ReLU (Primer / modded-nanogpt): strong small-scale baseline
            self.activation = lambda x: F.relu(x).square()
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

        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.load_balancing_loss_weight = config.load_balancing_loss_weight
        self.router_z_loss_weight = config.router_z_loss_weight
        self.router_type = getattr(config, 'moe_router_type', 'softmax')
        self.aux_free_bias = getattr(config, 'moe_aux_free_bias', False)
        self.bias_update_rate = getattr(config, 'moe_bias_update_rate', 0.001)

        # Router: learns which experts to use for each token
        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)

        # DeepSeek-V3 aux-loss-free balancing: a per-expert bias used only for
        # top-k SELECTION (never for output weighting), nudged toward even
        # expert load once per optimizer step (apply_aux_free_bias_updates in
        # the trainer). Forward only ACCUMULATES counts — mutating the bias in
        # forward would change routing between the checkpointing forward and
        # its backward recomputation and crash. Persistent so it survives export.
        if self.aux_free_bias:
            self.register_buffer('expert_bias', torch.zeros(self.num_experts))
            self.register_buffer('expert_load_acc', torch.zeros(self.num_experts), persistent=False)
        else:
            self.expert_bias = None
            self.expert_load_acc = None

        def make_expert():
            if config.activation == 'swiglu':
                return SwiGLU(config.d_model, config.d_ff, config.dropout)
            elif config.activation == 'geglu':
                return GeGLU(config.d_model, config.d_ff, config.dropout)
            elif config.activation == 'reglu':
                return ReGLU(config.d_model, config.d_ff, config.dropout)
            return StandardFFN(config.d_model, config.d_ff, config.activation, config.dropout)

        # Experts: create num_experts copies of the same FFN type
        self.experts = nn.ModuleList([make_expert() for _ in range(self.num_experts)])

        # DeepSeek-MoE style always-active shared expert
        self.shared_expert = make_expert() if getattr(config, 'moe_shared_expert', False) else None

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
        if self.router_type == 'sigmoid':
            # DeepSeek-V3: per-expert sigmoid affinities
            router_probs = torch.sigmoid(router_logits)
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        # Top-K selection — the aux-free bias shifts SELECTION only, never weights
        selection_scores = router_probs
        if self.expert_bias is not None:
            selection_scores = router_probs + self.expert_bias
        _, topk_indices = torch.topk(selection_scores, k=self.num_experts_per_token, dim=-1)
        topk_probs = router_probs.gather(-1, topk_indices)

        # Normalize topk probabilities (so they sum to 1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Aux-free balancing: only ACCUMULATE the load statistics here
        # (checkpoint recomputation may run this twice — a uniform x2 on the
        # counts does not change the sign-based update applied per step)
        if self.expert_load_acc is not None and self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_indices.flatten(), minlength=self.num_experts)
                self.expert_load_acc += counts.to(self.expert_load_acc.dtype)

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
            expert_weights = torch.zeros(token_expert_mask.sum(), device=x.device, dtype=x.dtype)
            for k in range(self.num_experts_per_token):
                mask_k = expert_mask[token_expert_mask, k]  # Tokens where this expert is in position k
                expert_weights[mask_k] = topk_probs[token_expert_mask, k][mask_k]

            # Add weighted expert output to final output
            output[token_expert_mask] += expert_weights.unsqueeze(-1) * expert_out

        # Always-active shared expert (added on top of routed experts)
        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        # Reshape back to (batch, seq_len, d_model)
        output = output.view(batch_size, seq_len, d_model)

        # Auxiliary losses (softmax router only — aux-free bias replaces them)
        if self.aux_free_bias:
            aux_loss = None
        else:
            aux_loss = self._compute_aux_loss(router_logits, router_probs, topk_indices)

        return output, aux_loss

    def apply_bias_update(self):
        """Apply one aux-loss-free balancing step (call once per optimizer step)."""
        if self.expert_bias is None:
            return
        with torch.no_grad():
            counts = self.expert_load_acc.float()
            if counts.sum() > 0:
                update = self.bias_update_rate * torch.sign(counts.mean() - counts)
                # compute in fp32 to survive a bf16 model cast
                self.expert_bias.copy_((self.expert_bias.float() + update).to(self.expert_bias.dtype))
            self.expert_load_acc.zero_()

    def _apply(self, fn, recurse=True):
        # Keep the routing bias in fp32 even when the model is cast to bf16 —
        # bf16 quantization would swallow the tiny per-step updates
        out = super()._apply(fn, recurse)
        if self.expert_bias is not None and self.expert_bias.dtype != torch.float32:
            self.expert_bias.data = self.expert_bias.data.float()
            self.expert_load_acc.data = self.expert_load_acc.data.float()
        return out

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
    'relu2': 'relu2',
    'gelu': 'gelu',
    'silu': 'silu',
    'leaky_relu': 'leaky_relu',
    'swiglu': SwiGLU,
    'geglu': GeGLU,
    'reglu': ReGLU
}
