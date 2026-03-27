"""
MAXIS Loss — Matryoshka Sampled Softmax Loss.

A fast approximation to full-vocabulary cross-entropy that avoids materialising
the [N × vocab_size] logit matrix.  Adapted from MaximusLLM by Yousef Rafat
(https://github.com/yousef-rafat/MaximusLLM, MIT licence).

Key ideas:
  1. Scout phase (no-grad): every 4th token uses a cheap 64-dim dot-product to
     find the top-k most-likely candidates in the vocabulary.
  2. Main loss: full-dim dot-products against the target + sampled candidates only.
  3. Ghost token: a synthetic logit that compensates for the unsampled vocabulary,
     keeping gradient magnitude consistent with true cross-entropy.
  4. Aux loss: same computation on the first `low_rank_dim` dimensions — acts as a
     Matryoshka regulariser that encourages compact representations.

Usage::

    loss_fn = MAXISLoss(model.lm_head.weight, vocab_size=model_config.vocab_size)
    # model must return hidden states, not logits:
    hidden, aux = model(x, return_hidden=True)
    loss = loss_fn(hidden[:, :-1].reshape(-1, d), targets[:, 1:].reshape(-1))

The returned loss is in the same units as cross-entropy (nats per token), so
perplexity can be computed as exp(loss) — but note it is an *approximation*, not
the true NLL.  Expect slightly lower values than CE at the same convergence point.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MAXISFunction(torch.autograd.Function):
    """
    Custom autograd Function for MAXIS loss.

    Avoids building the full [N, vocab_size] logit matrix by:
      - sampling n_candidates negatives per chunk via a cheap low-rank scout
      - adding a ghost logit to account for unsampled vocabulary mass
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,   # [N, d_model]
        embed_weight: torch.Tensor,    # [vocab_size, d_model]
        target_ids: torch.Tensor,      # [N]
        vocab_size: int,
        low_rank_dim: int,
        n_candidates: int,
        chunk_size: int,
        aux_weight: float,
    ) -> torch.Tensor:

        N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        h_f = hidden_states
        h_l = h_f[:, :low_rank_dim]
        w_low = embed_weight[:, :low_rank_dim]

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        saved_chunks = []

        with torch.no_grad():
            w_norm_sq = (embed_weight ** 2).sum(dim=-1).mean().item()
            w_low_norm_sq = (w_low ** 2).sum(dim=-1).mean().item()

        stride = 4
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            curr_h_f = h_f[i:end]
            curr_h_l = h_l[i:end]
            curr_t_ids = target_ids[i:end]

            # --- Scout: cheap low-rank candidate selection (no grad) ---
            with torch.no_grad():
                h_scouts = curr_h_l[::stride]
                scan_logits = torch.matmul(h_scouts, w_low.t())
                k_per_scout = max(1, n_candidates // max(1, chunk_size // stride))
                _, top_indices = torch.topk(scan_logits, k_per_scout, dim=1)
                top_indices = top_indices.reshape(-1)

            # --- Main (full-dim) loss ---
            w_f_pos = embed_weight[curr_t_ids]
            w_f_cand = embed_weight[top_indices]

            pos_sims = (curr_h_f * w_f_pos).sum(dim=-1, keepdim=True)
            neg_sims = torch.matmul(curr_h_f, w_f_cand.t())

            is_target = (top_indices.unsqueeze(0) == curr_t_ids.unsqueeze(1))
            neg_sims = neg_sims.masked_fill(is_target, float('-inf'))
            logits_m = torch.cat([pos_sims, neg_sims], dim=1).float()

            # Ghost token: approximates contribution of the unsampled vocabulary
            V_rem = vocab_size - top_indices.size(0) - 1
            full_dim = hidden_states.size(1)
            curr_h_sq = (curr_h_f ** 2).sum(dim=-1, keepdim=True).float()
            var_m = (curr_h_sq * w_norm_sq) / full_dim
            ghost_logits_m = math.log(max(1, V_rem)) + (var_m / 2.0)
            logits_m = torch.cat([logits_m, ghost_logits_m], dim=1)

            log_p_m = F.log_softmax(logits_m, dim=-1)
            loss_m = -log_p_m[:, 0].sum()

            # --- Auxiliary low-rank loss (Matryoshka regulariser) ---
            w_l_pos = w_f_pos[:, :low_rank_dim]
            w_l_cand = w_f_cand[:, :low_rank_dim]

            low_pos = (curr_h_l * w_l_pos).sum(dim=-1, keepdim=True)
            low_neg = torch.matmul(curr_h_l, w_l_cand.t())
            low_neg = low_neg.masked_fill(is_target, float('-inf'))
            logits_a = torch.cat([low_pos, low_neg], dim=1).float()

            curr_h_l_sq = (curr_h_l ** 2).sum(dim=-1, keepdim=True).float()
            var_a = (curr_h_l_sq * w_low_norm_sq) / low_rank_dim
            ghost_logits_a = math.log(max(1, V_rem)) + (var_a / 2.0)
            logits_a = torch.cat([logits_a, ghost_logits_a], dim=1)

            log_p_a = F.log_softmax(logits_a, dim=-1)
            loss_a = -log_p_a[:, 0].sum()

            total_loss += loss_m + aux_weight * loss_a

            saved_chunks.append((
                curr_h_f, curr_t_ids, top_indices,
                log_p_m.exp().to(dtype), log_p_a.exp().to(dtype),
                w_f_pos, w_f_cand, w_l_pos, w_l_cand,
                logits_m, logits_a,
            ))

        ctx.save_for_backward(hidden_states, embed_weight)
        ctx.aux_weight = aux_weight
        ctx.low_rank_dim = low_rank_dim
        ctx.saved_chunks = saved_chunks

        return total_loss / N

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, embed_weight = ctx.saved_tensors
        aux_weight = ctx.aux_weight

        N, _ = hidden_states.shape
        dtype = hidden_states.dtype

        grad_h = torch.zeros_like(hidden_states, dtype=torch.float32)
        grad_embed = torch.zeros_like(embed_weight, dtype=torch.float32)

        chunk_size = ctx.saved_chunks[0][0].shape[0]
        grad_embed_low = grad_embed[:, :ctx.low_rank_dim]

        for i, chunk in enumerate(ctx.saved_chunks):
            h_f, t_ids, top_idx, p_m, p_a, w_fp, w_fc, w_lp, w_lc, _l_m, _l_a = chunk

            w_fp = w_fp.to(dtype)
            w_fc = w_fc.to(dtype)
            w_lp = w_lp.to(dtype)
            w_lc = w_lc.to(dtype)

            # CE gradient: (softmax - one_hot_target)
            dz_m = p_m.float().clone()
            dz_m[:, 0] -= 1.0
            dz_m = dz_m * (grad_output.float() / N)

            dz_a = p_a.float().clone()
            dz_a[:, 0] -= 1.0
            dz_a = dz_a * (grad_output.float() / N) * aux_weight

            dz_m_dt = dz_m.to(dtype)
            dz_a_dt = dz_a.to(dtype)

            # Gradient w.r.t. hidden states
            gh = dz_m_dt[:, :1] * w_fp + torch.matmul(dz_m_dt[:, 1:-1], w_fc)
            ga = dz_a_dt[:, :1] * w_lp + torch.matmul(dz_a_dt[:, 1:-1], w_lc)
            gh[:, :ctx.low_rank_dim] += ga
            grad_h[i * chunk_size: i * chunk_size + h_f.shape[0]] = gh.float()

            # Gradient w.r.t. embedding weights
            h_f_float = h_f.float()
            grad_embed.index_add_(0, t_ids, dz_m[:, :1] * h_f_float)
            grad_embed.index_add_(0, top_idx, torch.matmul(dz_m_dt[:, 1:-1].t(), h_f).float())

            h_l_float = h_f_float[:, :ctx.low_rank_dim]
            grad_embed_low.index_add_(0, t_ids, dz_a[:, :1] * h_l_float)
            grad_embed_low.index_add_(0, top_idx,
                torch.matmul(dz_a_dt[:, 1:-1].t(), h_f[:, :ctx.low_rank_dim]).float())

        return grad_h.to(dtype), grad_embed.to(dtype), None, None, None, None, None, None


class MAXISLoss(nn.Module):
    """
    Matryoshka Sampled Softmax Loss.

    Args:
        embed_weight:   The model's output embedding matrix [vocab_size, d_model].
                        For weight-tied models this is the same tensor as the input
                        embedding, which is the typical setup in LLM-Lab.
        vocab_size:     True vocabulary size.  MUST match embed_weight.size(0).
        low_rank_dim:   Dimension used for the cheap scout pass and aux loss (default 64).
        n_candidates:   Number of negative candidates sampled per chunk (default 2048).
        chunk_size:     Tokens processed per inner loop iteration (default 32).
        aux_weight:     Weight of the Matryoshka auxiliary loss (default 0.2).
    """

    def __init__(
        self,
        embed_weight: torch.Tensor,
        vocab_size: int,
        low_rank_dim: int = 64,
        n_candidates: int = 2048,
        chunk_size: int = 32,
        aux_weight: float = 0.2,
    ):
        super().__init__()
        self.embed_weight = embed_weight
        self.vocab_size = vocab_size
        self.low_rank_dim = low_rank_dim
        self.n_candidates = n_candidates
        self.chunk_size = chunk_size
        self.aux_weight = aux_weight

        if low_rank_dim > embed_weight.size(1):
            raise ValueError(
                f"low_rank_dim ({low_rank_dim}) must be ≤ d_model ({embed_weight.size(1)})"
            )
        if vocab_size != embed_weight.size(0):
            raise ValueError(
                f"vocab_size ({vocab_size}) must match embed_weight.size(0) ({embed_weight.size(0)})"
            )

    def forward(self, hidden_states: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [N, d_model] — pre-lm_head hidden states (already flattened).
            target_ids:    [N]          — integer token targets.

        Returns:
            Scalar loss (approximate NLL per token).
        """
        return _MAXISFunction.apply(
            hidden_states,
            self.embed_weight,
            target_ids,
            self.vocab_size,
            self.low_rank_dim,
            self.n_candidates,
            self.chunk_size,
            self.aux_weight,
        )
