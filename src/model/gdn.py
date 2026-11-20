"""
Gated DeltaNet — linear attention with the gated delta rule.

The production pattern of Qwen3-Next / Qwen3.5 / Kimi Linear (2025-2026):
a fixed-size recurrent state updated with the delta rule (write new value,
erase the old value stored under the same key) plus a Mamba-2-style learned
decay gate. O(1) memory per token at inference — no KV cache.

The two rule implementations below are adapted from the MIT-licensed
reference in fla-org/flash-linear-attention (fla/ops/gated_delta_rule/naive.py,
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li), rewritten
without the einops dependency.

Shapes follow the fla convention:
    q, k: (B, T, H, K)   v: (B, T, H, V)   beta, g: (B, T, H)
    state: (B, H, K, V); g is a log-space decay (g <= 0, factor = exp(g))

doc_ids (B, T) optionally marks packed-document boundaries: the recurrent
state is reset whenever the document id changes, so packed documents are
exactly isolated (same guarantee the attention bricks get from their masks).

On CUDA, if the `fla` package (flash-linear-attention) is installed, its
Triton kernel is used for the chunkwise path — after a one-time numerical
parity check against the recurrent reference, so an API or semantics change
in fla can only ever cause a silent fallback, never wrong training.
Set LLMLAB_NO_FLA=1 to force the pure-PyTorch path.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def recurrent_gated_delta_rule(q, k, v, beta, g, scale=None,
                               initial_state=None, output_final_state=False,
                               doc_ids=None):
    """Sequential reference: exact, O(T) steps. Used for decode and as the
    ground truth the chunkwise version is tested against."""
    q, k, v = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v))
    beta, g = (x.transpose(1, 2).contiguous().to(torch.float32) for x in (beta, g))
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, dtype=torch.float32, device=v.device)
    S = torch.zeros(B, H, K, V, dtype=torch.float32, device=v.device)
    if initial_state is not None:
        S = initial_state.to(torch.float32)
    if scale is None:
        scale = K ** -0.5
    q = q * scale

    for i in range(T):
        if doc_ids is not None and i > 0:
            # new document starts here → wipe the carried state
            keep = (doc_ids[:, i] == doc_ids[:, i - 1]).view(B, 1, 1, 1)
            S = S * keep.to(S.dtype)
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i]
        S = S * g[:, :, i].exp()[..., None, None]
        # delta rule: replace the value currently stored under key k
        b_v = b_v - (S * b_k[..., None]).sum(-2)
        b_v = b_v * beta[:, :, i][..., None]
        S = S + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, S)

    return o.transpose(1, 2).contiguous(), (S if output_final_state else None)


def chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=64, scale=None,
                           initial_state=None, output_final_state=False,
                           doc_ids=None):
    """Chunkwise-parallel form (WY representation / UT transform with decay).

    Mathematically identical to the recurrent form; O(T/C) sequential steps
    of batched matmuls instead of O(T) — the trainable-speed path.

    Document isolation is enforced with exact zero masks (not by injecting
    -inf decays, which would wreck fp32 precision in the cumulative sums):
      - within a chunk, the pairwise decay matrix is restricted to same-doc
        pairs, which makes the UT-transform matrix block-diagonal per doc;
      - only tokens in the same doc as the incoming state may read it;
      - only tokens in the same doc as the chunk's last token write into the
        state carried to the next chunk, and the incoming state survives the
        chunk only if no boundary occurred inside it.
    """
    C = chunk_size
    if scale is None:
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

    # (B, H, L, ...) -> (B, H, n, C, ...)
    q = q.view(B, H, n, C, K)
    k = k.view(B, H, n, C, K)
    v = v.view(B, H, n, C, V)
    k_beta = k_beta.view(B, H, n, C, K)
    decay = g.view(B, H, n, C).cumsum(-1)  # within-chunk cumulative log-decay

    decay_exp = decay.exp()[..., None]
    # pairwise decay factors exp(cum_i - cum_j) for i >= j, tril-masked
    # before AND after exp so the upper region can't overflow
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()

    read_gate = carry_gate = state_carry = None
    if doc_ids is not None:
        d = doc_ids
        if pad > 0:  # pads inherit the last doc so they stay no-ops (k=0, beta=0)
            d = torch.cat([d, d[:, -1:].expand(-1, pad)], dim=1)
        d_c = d.view(B, 1, n, C)
        L_mask = L_mask * (d_c.unsqueeze(-1) == d_c.unsqueeze(-2)).to(L_mask.dtype)
        last_doc = d_c[..., -1]                                          # (B,1,n)
        state_doc = torch.cat([d_c[:, :, :1, 0], last_doc[:, :, :-1]], dim=2)
        read_gate = (d_c == state_doc.unsqueeze(-1)).to(torch.float32)   # may read incoming S
        carry_gate = (d_c == last_doc.unsqueeze(-1)).to(torch.float32)   # writes into carried S
        state_carry = (state_doc == last_doc).to(torch.float32)          # incoming S survives chunk
        decay_exp = decay_exp * read_gate.unsqueeze(-1)

    diag_mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=0)
    A = ((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(diag_mask, 0)
    # UT transform: attn = (I + A)^{-1}, a unit-lower-triangular solve
    # (replaces the O(C) forward-substitution Python loop of the reference)
    eye = torch.eye(C, dtype=torch.float32, device=q.device)
    attn = torch.linalg.solve_triangular(A + eye, eye.expand_as(A),
                                         upper=False, unitriangular=True)

    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    v = k_cumsum

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    strict_mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(n):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(strict_mask, 0)
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        if read_gate is not None:
            o_inter = o_inter * read_gate[:, :, i].unsqueeze(-1)
        o[:, :, i] = o_inter + attn_i @ v_new
        carry = (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
        if carry_gate is not None:
            carry = carry * carry_gate[:, :, i].unsqueeze(-1)
            S = S * decay[:, :, i, -1, None, None].exp() * state_carry[:, :, i, None, None]
        else:
            S = S * decay[:, :, i, -1, None, None].exp()
        S = S + (k_i * carry).transpose(-1, -2) @ v_new

    o = o.view(B, H, L, V)[:, :, :T]
    return o.transpose(1, 2).contiguous(), (S if output_final_state else None)


# ─── Optional fla (flash-linear-attention) Triton fast path ──────────────────

_fla_chunk_kernel = None
_fla_checked = False


def _get_fla_kernel():
    """Return fla's chunk_gated_delta_rule if it imports AND passes a one-time
    numerical parity check against our recurrent reference on this GPU.
    Any failure (missing package, API change, bad numerics) → None, silently
    falling back to the pure-PyTorch chunkwise path."""
    global _fla_chunk_kernel, _fla_checked
    if _fla_checked:
        return _fla_chunk_kernel
    _fla_checked = True
    if os.environ.get('LLMLAB_NO_FLA') or not torch.cuda.is_available():
        return None
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
    except Exception:
        return None
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            B, T, H, K = 2, 128, 2, 32
            q = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
            k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
            v = torch.randn(B, T, H, K, device='cuda')
            beta = torch.rand(B, T, H, device='cuda')
            g = F.logsigmoid(torch.rand(B, T, H, device='cuda'))
            ref_o, ref_S = recurrent_gated_delta_rule(
                q, k, v, beta, g, output_final_state=True)
            out = fla_chunk(q=q.bfloat16(), k=k.bfloat16(), v=v.bfloat16(),
                            g=g.float(), beta=beta.bfloat16(),
                            initial_state=None, output_final_state=True)
            o, S = out[0], out[1]
            d_o = (o.float() - ref_o).abs().max().item()
            d_S = (S.float() - ref_S).abs().max().item()
            # bf16 kernel vs fp32 reference: real mismatches (e.g. swapped
            # g/beta semantics after an fla API change) measure O(1)
            if d_o > 5e-2 or d_S > 5e-2:
                raise RuntimeError(f"parity check failed (o={d_o:.3f}, S={d_S:.3f})")
        _fla_chunk_kernel = fla_chunk
        print("[gdn] using fla Triton kernel (parity check passed)")
    except Exception as e:
        print(f"[gdn] fla kernel disabled ({e}) — using pure-PyTorch chunkwise path")
        _fla_chunk_kernel = None
    return _fla_chunk_kernel


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer, drop-in replacement for an attention brick.

    Matches the attention brick contract:
        forward(x, past_key_values=None, start_pos=0, doc_ids=None)
            -> (output, new_cache)
    The cache is a 1-tuple (state,) with state (B, H, K, V) — constant memory
    regardless of sequence length.
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.chunk_size = 64

        inner = self.n_heads * self.d_k
        self.q_proj = nn.Linear(self.d_model, inner, bias=False)
        self.k_proj = nn.Linear(self.d_model, inner, bias=False)
        self.v_proj = nn.Linear(self.d_model, inner, bias=False)
        # beta (write strength) and decay gate — per-head scalars per token
        self.b_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
        self.a_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
        # Mamba2-style decay parameterization: g = -exp(A_log) * softplus(a + dt_bias)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.n_heads).uniform_(1.0, 16.0)))
        dt = torch.exp(torch.empty(self.n_heads).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(dt)))
        # per-head output norm + SiLU output gate (Qwen3-Next style)
        self.o_norm = nn.RMSNorm(self.d_k, eps=config.norm_eps)
        self.g_proj = nn.Linear(self.d_model, inner, bias=False)
        self.w_o = nn.Linear(inner, self.d_model, bias=False)

        # attention-brick interface compatibility (never used by GDN)
        self.pos_encoding = None

    def _project(self, x):
        B, T, _ = x.shape
        q = F.normalize(self.q_proj(x).view(B, T, self.n_heads, self.d_k), p=2, dim=-1)
        k = F.normalize(self.k_proj(x).view(B, T, self.n_heads, self.d_k), p=2, dim=-1)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.b_proj(x))
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        return q, k, v, beta, g

    def forward(self, x, past_key_values=None, start_pos=0, doc_ids=None):
        B, T, _ = x.shape
        q, k, v, beta, g = self._project(x)

        state = past_key_values[0] if past_key_values is not None else None
        if past_key_values is not None or T == 1:
            # decode / short path: exact sequential recurrence
            o, new_state = recurrent_gated_delta_rule(
                q, k, v, beta, g, initial_state=state, output_final_state=True,
                doc_ids=doc_ids)
        else:
            fla_kernel = _get_fla_kernel() if (x.is_cuda and doc_ids is None) else None
            if fla_kernel is not None:
                out = fla_kernel(q=q, k=k, v=v, g=g, beta=beta.to(q.dtype),
                                 initial_state=state, output_final_state=True)
                o, new_state = out[0], out[1]
            else:
                o, new_state = chunk_gated_delta_rule(
                    q, k, v, beta, g, chunk_size=self.chunk_size,
                    initial_state=state, output_final_state=True, doc_ids=doc_ids)

        o = self.o_norm(o.to(x.dtype))
        o = o * F.silu(self.g_proj(x).view(B, T, self.n_heads, self.d_k))
        o = o.reshape(B, T, self.n_heads * self.d_k)
        return self.w_o(o), (new_state.detach(),)
