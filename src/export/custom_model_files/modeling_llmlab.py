"""
LLM-Lab Model Implementation for HuggingFace Transformers

This file is bundled with custom model exports to enable loading via
AutoModelForCausalLM.from_pretrained() with trust_remote_code=True.

Contains self-contained implementations of:
- TransformerLLM with various attention types (GQA, MHA, MQA, MLA)
- Various activations (SwiGLU, GeGLU, etc.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_llmlab import LLMLabConfig


# =============================================================================
# POSITIONAL ENCODINGS
# =============================================================================

class RoPE(nn.Module):
    """Rotary Position Embedding.

    The cos/sin tables are built lazily on first forward (on the input's
    device) rather than in __init__, so the module works with transformers'
    meta-device initialization (from_pretrained) and device_map loading.
    """
    def __init__(self, d_k: int, max_seq_len: int, fraction: float = 0.5):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.fraction = fraction
        self.cos = None
        self.sin = None

    def _build_tables(self, device):
        n_pairs = self.d_k // 2
        n_active = min(n_pairs, max(1, round(n_pairs * self.fraction)))
        # Match the training module's CPU-built float32 tables exactly. CUDA
        # trigonometry differs slightly and bf16 rounding can amplify this into
        # different logits after export. Keep lazy construction for meta loading.
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=n_active, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(n_pairs - n_active)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = theta.cos().to(device)
        self.sin = theta.sin().to(device)

    def forward(self, x_BTHD: torch.Tensor):
        """x: (batch, n_heads, seq_len, d_k)"""
        if self.cos is None or self.cos.device != x_BTHD.device:
            self._build_tables(x_BTHD.device)
        assert self.cos.size(0) >= x_BTHD.size(-2)
        cos = self.cos[None, None, :x_BTHD.size(-2), :]
        sin = self.sin[None, None, :x_BTHD.size(-2), :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_BTHD)


# =============================================================================
# ATTENTION MECHANISMS
# =============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for Grouped Query Attention"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class LightningIndexer(nn.Module):
    """DSA-style lightning indexer — mirrors training module exactly."""

    def __init__(self, d_model: int, n_heads: int = 4, d_index: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.d_index = d_index
        self.q_proj = nn.Linear(d_model, n_heads * d_index, bias=False)
        self.k_proj = nn.Linear(d_model, d_index, bias=False)
        self.w_proj = nn.Linear(d_model, n_heads, bias=False)

    def scores(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_index)
        k = self.k_proj(x)
        w = self.w_proj(x)
        qk = torch.einsum('bthd,bsd->bhts', q, k)
        return torch.einsum('bth,bhts->bts', w, F.relu(qk))


def _topk_sparse_mask(index_scores: torch.Tensor, top_k: int) -> torch.Tensor:
    B, T, _ = index_scores.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=index_scores.device))
    scores = index_scores.masked_fill(~causal, float('-inf'))
    k = min(top_k, T)
    topk_idx = scores.topk(k, dim=-1).indices
    mask = torch.zeros(B, T, T, dtype=torch.bool, device=index_scores.device)
    mask.scatter_(-1, topk_idx, True)
    mask = mask & causal
    # diagonal always included (mirrors training-time mask exactly)
    mask.diagonal(dim1=-2, dim2=-1).fill_(True)
    return mask.unsqueeze(1)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (also covers MHA and MQA as special cases).

    Mirrors the training-time modules exactly:
    - MHA: num_kv_heads == num_heads, no QK-norm
    - MQA: num_kv_heads == 1, no QK-norm
    - GQA: num_kv_heads from config, QK-norm on each head
    """
    def __init__(self, config: LLMLabConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_type = config.attention_type
        if self.attention_type == "mha":
            self.num_kv_heads = self.num_heads
        elif self.attention_type == "mqa":
            self.num_kv_heads = 1
        else:
            self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.w_o = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # QK normalization (training code only applies it to GQA)
        if self.attention_type == "gqa":
            if config.norm_type == "rmsnorm":
                self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
                self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps)
                self.k_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.pos_encoding = None

        # Qwen3-Next-style sigmoid output gate
        if getattr(config, 'use_attention_gate', False):
            self.attn_gate = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        else:
            self.attn_gate = None

        # DSA-style sparse attention
        if getattr(config, 'use_sparse_attention', False):
            self.indexer = LightningIndexer(
                config.hidden_size,
                n_heads=getattr(config, 'sparse_indexer_heads', 4),
                d_index=getattr(config, 'sparse_indexer_dim', 32),
            )
            self.sparse_top_k = getattr(config, 'sparse_top_k', 512)
        else:
            self.indexer = None

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.pos_encoding is not None:
            q = self.pos_encoding(q)
            k = self.pos_encoding(k)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        if self.indexer is not None:
            sparse_mask = _topk_sparse_mask(self.indexer.scores(x), self.sparse_top_k)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sparse_mask, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        if self.attn_gate is not None:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
        return self.w_o(attn_output)


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) - DeepSeek style"""
    def __init__(self, config: LLMLabConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.d_latent = config.d_latent or (config.hidden_size // 4)
        self.d_rope_latent = config.d_rope_latent or self.head_dim
        self.dropout = config.attention_dropout
        self.positional_encoding = config.positional_encoding

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.kv_down = nn.Linear(self.hidden_size, self.d_latent, bias=False)

        if self.positional_encoding in ["rope", "yarn"]:
            self.k_rope_proj = nn.Linear(self.d_latent, self.num_heads * self.d_rope_latent, bias=False)
            self.k_nope_proj = nn.Linear(self.d_latent, self.num_heads * (self.head_dim - self.d_rope_latent), bias=False)
        else:
            self.k_proj = nn.Linear(self.d_latent, self.num_heads * self.head_dim, bias=False)

        self.v_proj = nn.Linear(self.d_latent, self.num_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if config.norm_type == "rmsnorm":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        else:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps)

        self.pos_encoding = None

        # Qwen3-Next-style sigmoid output gate
        if getattr(config, 'use_attention_gate', False):
            self.attn_gate = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        else:
            self.attn_gate = None

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        latent = self.kv_down(x)
        v = self.v_proj(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)

        if self.positional_encoding in ["rope", "yarn"]:
            k_rope = self.k_rope_proj(latent).view(batch_size, seq_len, self.num_heads, self.d_rope_latent)
            k_nope = self.k_nope_proj(latent).view(batch_size, seq_len, self.num_heads, self.head_dim - self.d_rope_latent)

            q = self.q_norm(q)
            q = q.transpose(1, 2)
            k_rope = k_rope.transpose(1, 2)
            k_nope = k_nope.transpose(1, 2)
            v = v.transpose(1, 2)

            if self.pos_encoding is not None:
                q_rope = q[..., :self.d_rope_latent]
                q_nope = q[..., self.d_rope_latent:]
                q_rope = self.pos_encoding(q_rope)
                k_rope = self.pos_encoding(k_rope)
                q = torch.cat([q_rope, q_nope], dim=-1)

            k = torch.cat([k_rope, k_nope], dim=-1)
            k = k.transpose(1, 2)
            k = self.k_norm(k)
            k = k.transpose(1, 2)
        else:
            k = self.k_proj(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        if self.attn_gate is not None:
            attn_output = attn_output * torch.sigmoid(self.attn_gate(x))
        return self.w_o(attn_output)


# =============================================================================
# FEED-FORWARD NETWORKS
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation"""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GeGLU(nn.Module):
    """GeGLU activation"""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.gelu(self.gate_proj(x)) * self.up_proj(x)))


class ReGLU(nn.Module):
    """ReGLU activation"""
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(self.dropout(F.relu(self.gate_proj(x)) * self.up_proj(x)))


class StandardFFN(nn.Module):
    """Standard FFN with configurable activation"""
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "relu2":
            self.activation = lambda x: F.relu(x).square()
        elif activation == "silu":
            self.activation = F.silu
        elif activation == "leaky_relu":
            self.activation = lambda x: F.leaky_relu(x, 0.01)
        else:
            self.activation = F.gelu

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


def _make_expert(config: LLMLabConfig):
    """Build a single FFN matching the training-time expert layout"""
    act = config.hidden_act
    if act == "swiglu":
        return SwiGLU(config.hidden_size, config.intermediate_size, config.dropout)
    if act == "geglu":
        return GeGLU(config.hidden_size, config.intermediate_size, config.dropout)
    if act == "reglu":
        return ReGLU(config.hidden_size, config.intermediate_size, config.dropout)
    return StandardFFN(config.hidden_size, config.intermediate_size, act, config.dropout)


class MoEFFN(nn.Module):
    """Mixture of Experts FFN (inference version of the training MoEFFN).

    Same parameter layout as training: `router` + `experts.{i}` so exported
    weights load 1:1. Auxiliary losses are training-only and omitted here.
    """
    def __init__(self, config: LLMLabConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.router_type = getattr(config, 'moe_router_type', 'softmax')
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        if getattr(config, 'moe_aux_free_bias', False):
            self.register_buffer('expert_bias', torch.zeros(self.num_experts))
        else:
            self.expert_bias = None
        self.experts = nn.ModuleList([_make_expert(config) for _ in range(self.num_experts)])
        self.shared_expert = _make_expert(config) if getattr(config, 'moe_shared_expert', False) else None

    def forward(self, x):
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)

        router_logits = self.router(x_flat)
        if self.router_type == 'sigmoid':
            router_probs = torch.sigmoid(router_logits)
        else:
            router_probs = F.softmax(router_logits, dim=-1)
        selection_scores = router_probs if self.expert_bias is None else router_probs + self.expert_bias
        _, topk_indices = torch.topk(selection_scores, k=self.num_experts_per_token, dim=-1)
        topk_probs = router_probs.gather(-1, topk_indices)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            expert_mask = (topk_indices == expert_idx)
            token_mask = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            expert_out = self.experts[expert_idx](x_flat[token_mask])
            expert_weights = torch.zeros(token_mask.sum(), device=x.device, dtype=x.dtype)
            for k in range(self.num_experts_per_token):
                mask_k = expert_mask[token_mask, k]
                expert_weights[mask_k] = topk_probs[token_mask, k][mask_k]
            output[token_mask] += expert_weights.unsqueeze(-1) * expert_out

        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        return output.view(batch_size, seq_len, hidden)


# =============================================================================
# GATED DELTANET (hybrid linear layers)
# =============================================================================

def _recurrent_gated_delta_rule(q, k, v, beta, g, initial_state=None):
    """Sequential gated delta rule (adapted from MIT-licensed fla-org reference).

    q, k: (B, T, H, K) — L2-normalized; v: (B, T, H, V); beta, g: (B, T, H),
    g in log space. Returns output (B, T, H, V).
    """
    q, k, v = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v))
    beta, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (beta, g))
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, dtype=torch.float32, device=v.device)
    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=v.device)
    if initial_state is not None:
        S = initial_state.to(torch.float32)
    q = q * (K ** -0.5)

    for i in range(T):
        b_q, b_k, b_v = q[:, :, i], k[:, :, i], v[:, :, i]
        S = S * g[:, :, i].exp()[..., None, None]
        b_v = b_v - (S * b_k[..., None]).sum(-2)
        b_v = b_v * beta[:, :, i][..., None]
        S = S + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, S)

    return o.transpose(1, 2).contiguous()


def _chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=64):
    """Chunkwise-parallel gated delta rule (WY/UT transform) — identical to the
    sequential form, O(T/C) sequential steps. Used for multi-token prefill."""
    C = chunk_size
    scale = q.shape[-1] ** -0.5

    q, k, v = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v))
    beta, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (beta, g))

    T = q.shape[-2]
    pad = (C - T % C) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))

    B, H, L, K = q.shape
    V = v.shape[-1]
    n = L // C
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q = q.view(B, H, n, C, K)
    k = k.view(B, H, n, C, K)
    v = v.view(B, H, n, C, V)
    k_beta = k_beta.view(B, H, n, C, K)
    decay = g.view(B, H, n, C).cumsum(-1)

    decay_exp = decay.exp()[..., None]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()

    diag_mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)
    A = ((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(diag_mask, 0)
    # UT transform: attn = (I + A)^{-1}, a unit-lower-triangular solve
    eye = torch.eye(C, dtype=torch.float32, device=q.device)
    attn = torch.linalg.solve_triangular(A + eye, eye.expand_as(A),
                                         upper=False, unitriangular=True)

    v = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)

    S = k.new_zeros(B, H, K, V)
    o = torch.zeros_like(v)
    strict_mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(n):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(strict_mask, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ S
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn_i @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + \
            (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new

    o = o.view(B, H, L, V)[:, :, :T]
    return o.transpose(1, 2).contiguous()


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer — mirrors the training-time module exactly."""

    def __init__(self, config: LLMLabConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        inner = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, inner, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.g_proj = nn.Linear(self.hidden_size, inner, bias=False)
        self.w_o = nn.Linear(inner, self.hidden_size, bias=False)
        self.pos_encoding = None

    def forward(self, x):
        B, T, _ = x.shape
        q = F.normalize(self.q_proj(x).view(B, T, self.num_heads, self.head_dim), p=2, dim=-1)
        k = F.normalize(self.k_proj(x).view(B, T, self.num_heads, self.head_dim), p=2, dim=-1)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.b_proj(x))
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)

        if T > 1:
            o = _chunk_gated_delta_rule(q, k, v, beta, g)
        else:
            o = _recurrent_gated_delta_rule(q, k, v, beta, g)
        o = self.o_norm(o.to(x.dtype))
        o = o * F.silu(self.g_proj(x).view(B, T, self.num_heads, self.head_dim))
        return self.w_o(o.reshape(B, T, self.num_heads * self.head_dim))


def _is_linear_layer(config, layer_idx: int) -> bool:
    pattern = getattr(config, 'hybrid_pattern', None)
    if not pattern:
        return False
    lin, att = (int(p) for p in pattern.split(':'))
    return (layer_idx % (lin + att)) < lin


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, config: LLMLabConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Token mixer: attention, or Gated DeltaNet on hybrid linear layers
        if _is_linear_layer(config, layer_idx):
            self.attention = GatedDeltaNet(config)
        elif config.attention_type == "mla":
            self.attention = MultiHeadLatentAttention(config)
        else:
            self.attention = GroupedQueryAttention(config)

        # Feed-forward: MoE on this layer if enabled (mirrors training selection)
        moe_layers = getattr(config, "moe_layers", None)
        use_moe_this_layer = getattr(config, "use_moe", False) and (
            moe_layers is None or layer_idx in moe_layers
        )
        if use_moe_this_layer:
            self.feed_forward = MoEFFN(config)
        else:
            self.feed_forward = _make_expert(config)

        # Normalization
        if config.norm_type == "rmsnorm":
            self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
            self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


# =============================================================================
# MAIN MODEL
# =============================================================================

class LLMLabPreTrainedModel(PreTrainedModel):
    """Base class for LLM-Lab models"""
    config_class = LLMLabConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class LLMLabForCausalLM(LLMLabPreTrainedModel):
    """LLM-Lab Transformer for Causal Language Modeling"""

    # Dict form required by transformers >= 5 (target param -> source param)
    _tied_weights_keys = {"lm_head.weight": "token_embedding.weight"}

    def __init__(self, config: LLMLabConfig):
        super().__init__(config)
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Positional encoding
        if config.positional_encoding in ["rope", "yarn"]:
            self.pos_encoding = RoPE(config.head_dim, config.max_position_embeddings,
                                     fraction=getattr(config, 'rope_fraction', 0.5))
        else:
            self.pos_encoding = None

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])

        # Setup positional encoding for attention layers
        if self.pos_encoding is not None:
            nope_every = getattr(config, 'nope_every', None)
            for i, block in enumerate(self.transformer_blocks):
                if nope_every and (i + 1) % nope_every == 0:
                    continue
                block.attention.pos_encoding = self.pos_encoding

        # Final normalization
        if config.norm_type == "rmsnorm":
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            x = self.token_embedding(input_ids) * math.sqrt(self.config.hidden_size)
        else:
            x = inputs_embeds * math.sqrt(self.config.hidden_size)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final norm and output
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
