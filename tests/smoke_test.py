"""
Smoke tests for LLM-Lab.

Sections:
  1.  Forward pass: all attention × pos-encoding × sliding window combos
  2.  KV cache correctness: single-step cached decode == full forward logits
  3.  Multi-step KV cache: 5 sequential decode steps == full forward
  4.  Training backward: gradients flow through all attention types
  5.  DPO/PPO/GRPO tuple unpacking + backward
  6.  SFT forward + ignore_index=-100
  7.  Mamba2 forward + use_cache shape
  8.  MoE forward: aux_loss non-None + backward
  9.  Activation functions: swiglu / geglu / reglu / gelu / relu
  10. Norm types: rmsnorm / layernorm
  11. Sliding window + KV cache: no crash, correct sliding truncation
  12. ALiBi decode: verify bias shape for every attention type
  13. Gradient checkpointing: backward works with use_checkpoint=True
  14. DPO loss function: dpo_loss() returns finite loss + correct accuracy sign
  15. Sliding window mask: verify out-of-window positions are masked
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

from config import ModelConfig
from model.factory import build_model

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16
VOCAB  = 256
SEQ    = 16
BATCH  = 2

_PASS = 0
_FAIL = 0


def ok(name):
    global _PASS
    _PASS += 1
    print(f"  [PASS] {name}")


def fail(name, err):
    global _FAIL
    _FAIL += 1
    print(f"  [FAIL] {name}: {err}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tiny_config(
    attention_type="mha",
    positional_encoding="rope",
    sliding_window=None,
    n_kv_heads=None,
    use_moe=False,
    activation="swiglu",
    norm_type="rmsnorm",
):
    kwargs = dict(
        tokenizer_name="Qwen/Qwen2.5-0.5B",
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=VOCAB,
        max_seq_len=64,
        dropout=0.0,
        d_ff=128,
        attention_type=attention_type,
        positional_encoding=positional_encoding,
        sliding_window=sliding_window,
        norm_type=norm_type,
        activation=activation,
        use_moe=use_moe,
    )
    if use_moe:
        kwargs.update(num_experts=4, num_experts_per_token=2)
    if attention_type == "gqa":
        kwargs["n_kv_heads"] = n_kv_heads or 2
    if attention_type == "mqa":
        kwargs["n_kv_heads"] = 1
    if attention_type == "mla":
        kwargs["d_latent"] = 16
        kwargs["d_rope_latent"] = 16
    return ModelConfig(**kwargs)


def make_model(cfg):
    model = build_model(cfg)
    return model.to(device=DEVICE, dtype=DTYPE)


def rand_ids(batch=BATCH, seq=SEQ):
    return torch.randint(0, VOCAB, (batch, seq), device=DEVICE)


# ---------------------------------------------------------------------------
# 1. Forward pass — all combos
# ---------------------------------------------------------------------------
ATTN_TYPES    = ["mha", "mqa", "gqa", "mla"]
POS_ENCODINGS = ["rope", "yarn", "alibi", "sinusoidal", "learned"]


def test_forward_all_combos():
    print("\n[1] Forward pass — attention × positional encoding × sliding window")
    combos = []
    for attn in ATTN_TYPES:
        for pos in POS_ENCODINGS:
            if attn == "mla" and pos == "alibi":
                continue  # MLA+ALiBi not supported
            combos.append((attn, pos, None))
            combos.append((attn, pos, SEQ // 2))  # sliding window

    for attn, pos, sw in combos:
        name = f"{attn}+{pos}" + (f"+sw{sw}" if sw else "")
        try:
            cfg   = tiny_config(attention_type=attn, positional_encoding=pos, sliding_window=sw)
            model = make_model(cfg)
            x     = rand_ids()
            with torch.no_grad():
                out = model(x)
            logits = out[0]
            assert logits.shape == (BATCH, SEQ, VOCAB), f"shape={logits.shape}"
            assert torch.isfinite(logits).all(), "non-finite logits"
            del model
            ok(name)
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 2. KV cache — single-step correctness
# ---------------------------------------------------------------------------
def test_kv_cache_single_step():
    print("\n[2] KV cache — single-step cached decode == full forward")
    for attn in ATTN_TYPES:
        for pos in ["rope", "yarn", "alibi", "sinusoidal", "learned"]:
            if attn == "mla" and pos == "alibi":
                continue
            name = f"{attn}+{pos}"
            try:
                cfg   = tiny_config(attention_type=attn, positional_encoding=pos)
                model = make_model(cfg)
                model.eval()

                prompt  = rand_ids(batch=1, seq=8)
                new_tok = rand_ids(batch=1, seq=1)
                full_in = torch.cat([prompt, new_tok], dim=1)  # (1,9)

                with torch.no_grad():
                    logits_full = model(full_in)[0][0, 8, :]  # pos 8

                    out_prefill  = model(prompt, use_cache=True)
                    assert len(out_prefill) == 3, "prefill must return 3-tuple"
                    past_kv = out_prefill[2]

                    out_cached     = model(new_tok, past_key_values=past_kv, use_cache=True)
                    logits_cached  = out_cached[0][0, 0, :]

                diff = (logits_full - logits_cached).abs().max().item()
                assert diff < 0.1, f"max_diff={diff:.4f}"
                ok(f"{name} (diff={diff:.5f})")
                del model
            except Exception as e:
                fail(name, e)


# ---------------------------------------------------------------------------
# 3. KV cache — multi-step (5 tokens) correctness
# ---------------------------------------------------------------------------
def test_kv_cache_multistep():
    print("\n[3] KV cache — 5-token sequential decode == full forward")
    STEPS = 5
    P     = 8  # prompt length

    for attn in ATTN_TYPES:
        for pos in ["rope", "yarn", "alibi", "sinusoidal", "learned"]:
            if attn == "mla" and pos == "alibi":
                continue
            name = f"{attn}+{pos}"
            try:
                cfg   = tiny_config(attention_type=attn, positional_encoding=pos)
                model = make_model(cfg)
                model.eval()

                # Build one sequence long enough for prefill + decode
                all_tokens = rand_ids(batch=1, seq=P + STEPS)

                with torch.no_grad():
                    # Full forward on entire sequence
                    logits_full = model(all_tokens)[0]  # (1, P+STEPS, V)

                    # Prefill
                    out = model(all_tokens[:, :P], use_cache=True)
                    past_kv = out[2]

                    # Decode one token at a time
                    max_diff = 0.0
                    for step in range(STEPS):
                        tok = all_tokens[:, P + step : P + step + 1]
                        out = model(tok, past_key_values=past_kv, use_cache=True)
                        cached_logits   = out[0][0, 0, :]
                        expected_logits = logits_full[0, P + step, :]
                        diff = (cached_logits - expected_logits).abs().max().item()
                        max_diff = max(max_diff, diff)
                        past_kv = out[2]

                assert max_diff < 0.1, f"max_diff={max_diff:.4f} over {STEPS} steps"
                ok(f"{name} (max_diff={max_diff:.5f})")
                del model
            except Exception as e:
                fail(name, e)


# ---------------------------------------------------------------------------
# 4. Training backward — all attention types
# ---------------------------------------------------------------------------
def test_backward_all_attn():
    print("\n[4] Training backward — gradients flow for all attention types")
    for attn in ATTN_TYPES:
        name = f"{attn}+rope"
        try:
            cfg   = tiny_config(attention_type=attn)
            model = make_model(cfg)
            model.train()

            x       = rand_ids()
            targets = rand_ids()
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
            loss.backward()

            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            assert grad_norms, "no gradients"
            assert max(grad_norms) > 0, "all-zero gradients"
            ok(f"{name} (max_grad={max(grad_norms):.4f})")
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 5. DPO / PPO / GRPO tuple unpacking + backward
# ---------------------------------------------------------------------------
def test_rlhf_tuple_unpacking():
    print("\n[5] RLHF tuple unpacking (DPO/PPO/GRPO pattern) + backward")
    for attn in ["mha", "gqa"]:
        name = f"{attn}+rope"
        try:
            cfg   = tiny_config(attention_type=attn)
            model = make_model(cfg)
            model.train()

            x = rand_ids(batch=1, seq=12)
            logits, _ = model(x)  # the pattern that was broken
            assert logits.shape == (1, 12, VOCAB)

            resp_logits = logits[0, 5:11, :]
            resp_ids    = rand_ids(batch=1, seq=6)[0]
            lp   = F.log_softmax(resp_logits, dim=-1)
            tlp  = lp.gather(1, resp_ids.unsqueeze(1)).squeeze(1)
            loss = -tlp.mean()
            loss.backward()
            ok(name)
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 6. SFT forward — cross-entropy with ignore_index=-100
# ---------------------------------------------------------------------------
def test_sft_forward():
    print("\n[6] SFT forward — ignore_index=-100 masking")
    for attn in ["mha", "gqa"]:
        name = f"{attn}+rope"
        try:
            cfg   = tiny_config(attention_type=attn)
            model = make_model(cfg)
            model.train()

            x = rand_ids()
            y = rand_ids()
            y[:, :SEQ // 2] = -100  # mask prompt half

            logits, aux = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1), ignore_index=-100)
            if aux is not None:
                loss = loss + aux
            loss.backward()
            assert loss.item() > 0
            ok(name)
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 7. Mamba2 forward + use_cache shape
# ---------------------------------------------------------------------------
def test_mamba2_forward():
    print("\n[7] Mamba2 forward + use_cache shape")
    try:
        cfg = ModelConfig(
            model_architecture="mamba2",
            tokenizer_name="Qwen/Qwen2.5-0.5B",
            d_model=64, n_layers=2, vocab_size=VOCAB,
            max_seq_len=64, state_size=16, expand_factor=2,
            headdim=16, chunk_size=8,
        )
        model = make_model(cfg)
        model.eval()
        x = rand_ids(batch=1, seq=SEQ)

        with torch.no_grad():
            out = model(x)
        assert out[0].shape == (1, SEQ, VOCAB), f"shape={out[0].shape}"

        out_cached = model(x, use_cache=True)
        assert len(out_cached) == 3, "use_cache should return 3-tuple"
        ok("mamba2 forward + use_cache shape")
        del model
    except Exception as e:
        fail("mamba2", e)


# ---------------------------------------------------------------------------
# 8. MoE forward: aux_loss non-None + backward
# ---------------------------------------------------------------------------
def test_moe_forward():
    print("\n[8] MoE forward — aux_loss returned and backward works")
    for attn in ["mha", "gqa"]:
        name = f"moe+{attn}"
        try:
            cfg   = tiny_config(attention_type=attn, use_moe=True)
            model = make_model(cfg)
            model.train()

            x       = rand_ids()
            targets = rand_ids()
            logits, aux_loss = model(x)

            assert logits.shape == (BATCH, SEQ, VOCAB)
            assert aux_loss is not None, "MoE must return aux_loss"
            assert aux_loss.item() > 0, "aux_loss should be positive"

            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1)) + aux_loss
            loss.backward()

            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            assert max(grad_norms) > 0
            ok(f"{name} (aux_loss={aux_loss.item():.4f})")
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 9. Activation functions
# ---------------------------------------------------------------------------
def test_activation_functions():
    print("\n[9] Activation functions — forward + backward")
    activations = ["swiglu", "geglu", "reglu", "gelu", "relu"]
    for act in activations:
        try:
            cfg   = tiny_config(activation=act)
            model = make_model(cfg)
            model.train()
            x = rand_ids()
            logits, _ = model(x)
            assert logits.shape == (BATCH, SEQ, VOCAB)
            F.cross_entropy(logits.view(-1, VOCAB), rand_ids().view(-1)).backward()
            ok(act)
            del model
        except Exception as e:
            fail(act, e)


# ---------------------------------------------------------------------------
# 10. Norm types
# ---------------------------------------------------------------------------
def test_norm_types():
    print("\n[10] Norm types — rmsnorm / layernorm")
    for norm in ["rmsnorm", "layernorm"]:
        try:
            cfg   = tiny_config(norm_type=norm)
            model = make_model(cfg)
            model.train()
            x = rand_ids()
            logits, _ = model(x)
            assert logits.shape == (BATCH, SEQ, VOCAB)
            F.cross_entropy(logits.view(-1, VOCAB), rand_ids().view(-1)).backward()
            ok(norm)
            del model
        except Exception as e:
            fail(norm, e)


# ---------------------------------------------------------------------------
# 11. Sliding window + KV cache: truncation works, logits finite
# ---------------------------------------------------------------------------
def test_sliding_window_kv_cache():
    print("\n[11] Sliding window + KV cache — truncation + no crash")
    SW = 6  # window
    P  = 8  # prompt (> SW, so truncation will happen during decode)

    for attn in ["mha", "mqa", "gqa"]:  # mla tested separately
        name = f"{attn}+rope+sw{SW}"
        try:
            cfg   = tiny_config(attention_type=attn, sliding_window=SW)
            model = make_model(cfg)
            model.eval()

            prompt = rand_ids(batch=1, seq=P)

            with torch.no_grad():
                out_prefill = model(prompt, use_cache=True)
                assert len(out_prefill) == 3

                past_kv = out_prefill[2]
                # KV should be capped at SW after prefill (P > SW)
                # Actually: prefill doesn't truncate (no past_key_values on prefill)
                # Truncation only happens during decode when past_key_values is passed

                # Decode 4 tokens
                for _ in range(4):
                    tok = rand_ids(batch=1, seq=1)
                    out = model(tok, past_key_values=past_kv, use_cache=True)
                    assert torch.isfinite(out[0]).all(), "non-finite logits"
                    past_kv = out[2]
                    # After truncation, KV should be ≤ SW
                    for layer_kv in past_kv:
                        assert layer_kv[0].size(2) <= SW, \
                            f"KV len {layer_kv[0].size(2)} > window {SW}"

            ok(name)
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 12. ALiBi decode bias — shape check for all attn types
# ---------------------------------------------------------------------------
def test_alibi_decode_bias_shape():
    print("\n[12] ALiBi decode bias — all attention types")
    for attn in ["mha", "mqa", "gqa"]:  # MLA+ALiBi not supported
        name = f"{attn}+alibi"
        try:
            cfg   = tiny_config(attention_type=attn, positional_encoding="alibi")
            model = make_model(cfg)
            model.eval()

            prompt  = rand_ids(batch=1, seq=8)
            new_tok = rand_ids(batch=1, seq=1)

            with torch.no_grad():
                out_prefill   = model(prompt, use_cache=True)
                past_kv       = out_prefill[2]
                out_decode    = model(new_tok, past_key_values=past_kv, use_cache=True)
                logits_decode = out_decode[0]

            assert logits_decode.shape == (1, 1, VOCAB)
            assert torch.isfinite(logits_decode).all()
            ok(name)
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 13. Gradient checkpointing — backward works with use_checkpoint=True
# ---------------------------------------------------------------------------
def test_gradient_checkpointing():
    print("\n[13] Gradient checkpointing — backward with use_checkpoint=True")
    for attn in ["mha", "gqa"]:
        name = f"{attn}+rope+checkpoint"
        try:
            cfg   = tiny_config(attention_type=attn)
            model = make_model(cfg)
            model.train()

            x       = rand_ids()
            targets = rand_ids()
            logits, _ = model(x, use_checkpoint=True)
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
            loss.backward()

            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            assert max(grad_norms) > 0
            ok(f"{name} (max_grad={max(grad_norms):.4f})")
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 14. DPO loss function — synthetic log-probs
# ---------------------------------------------------------------------------
def test_dpo_loss_function():
    print("\n[14] DPO loss function — correctness with synthetic log-probs")
    from training.dpo_train import dpo_loss

    # Case 1: policy clearly prefers chosen → logit > 0 → accuracy = 1
    pc = torch.tensor([-1.0])   # policy chosen log-prob
    pr = torch.tensor([-3.0])   # policy rejected log-prob  (chosen >> rejected)
    rc = torch.tensor([-1.5])   # ref chosen
    rr = torch.tensor([-1.5])   # ref rejected (ref is neutral)

    try:
        loss, acc = dpo_loss(pc, pr, rc, rr, beta=0.1)
        assert torch.isfinite(loss), f"loss not finite: {loss}"
        assert acc.item() == 1.0, f"expected accuracy=1.0, got {acc.item()}"
        ok(f"dpo_loss chosen>rejected (loss={loss.item():.4f}, acc={acc.item():.1f})")
    except Exception as e:
        fail("dpo_loss chosen>rejected", e)

    # Case 2: policy prefers rejected (bad) → logit < 0 → accuracy = 0
    pc2 = torch.tensor([-3.0])
    pr2 = torch.tensor([-1.0])

    try:
        loss2, acc2 = dpo_loss(pc2, pr2, rc, rr, beta=0.1)
        assert torch.isfinite(loss2)
        assert acc2.item() == 0.0, f"expected accuracy=0.0, got {acc2.item()}"
        assert loss2.item() > loss.item(), "loss should be higher when policy is wrong"
        ok(f"dpo_loss rejected>chosen (loss={loss2.item():.4f}, acc={acc2.item():.1f})")
    except Exception as e:
        fail("dpo_loss rejected>chosen", e)


# ---------------------------------------------------------------------------
# 15. Sliding window mask correctness — out-of-window positions masked
# ---------------------------------------------------------------------------
def test_sliding_window_mask():
    print("\n[15] Sliding window mask — out-of-window positions are False")
    from model.bricks import _create_sliding_window_mask

    seq_len = 10
    window  = 3

    try:
        mask = _create_sliding_window_mask(seq_len, window, DEVICE)
        # mask shape: (1, 1, seq_len, seq_len), bool, True=allowed
        assert mask.shape == (1, 1, seq_len, seq_len)

        # Causal: upper triangle must be False
        for q in range(seq_len):
            for k in range(q + 1, seq_len):
                assert not mask[0, 0, q, k].item(), f"future pos ({q},{k}) should be masked"

        # Sliding window: positions too far back must be False
        for q in range(seq_len):
            for k in range(max(0, q - window) - 1, -1, -1):
                if k < q - window:
                    assert not mask[0, 0, q, k].item(), \
                        f"pos ({q},{k}) outside window={window} should be masked"

        # Within window + causal: must be True
        for q in range(seq_len):
            for k in range(max(0, q - window), q + 1):
                assert mask[0, 0, q, k].item(), \
                    f"pos ({q},{k}) within window={window} should be allowed"

        ok(f"window={window} seq_len={seq_len}")
    except Exception as e:
        fail("sliding_window_mask", e)


# ---------------------------------------------------------------------------
# 16. use_cache=False backward compat — still returns 2-tuple
# ---------------------------------------------------------------------------
def test_cache_backward_compat():
    print("\n[16] Backward compat — model() returns 2-tuple without use_cache")
    try:
        cfg   = tiny_config()
        model = make_model(cfg)
        model.eval()
        x = rand_ids(batch=1)
        with torch.no_grad():
            out = model(x)
        assert len(out) == 2, f"Expected 2-tuple, got {len(out)}-tuple"
        logits, aux = out
        assert logits.shape == (1, SEQ, VOCAB)
        assert aux is None
        ok("2-tuple without use_cache")
        del model
    except Exception as e:
        fail("backward_compat", e)


# ---------------------------------------------------------------------------
# 17. KV cache — batch size > 1 (prefill only, no decode)
# ---------------------------------------------------------------------------
def test_kv_cache_batched_prefill():
    print("\n[17] KV cache — batch > 1 prefill returns correct shapes")
    for attn in ["mha", "gqa"]:
        name = f"{attn}+rope+batch{BATCH}"
        try:
            cfg   = tiny_config(attention_type=attn)
            model = make_model(cfg)
            model.eval()
            x = rand_ids(batch=BATCH, seq=8)

            with torch.no_grad():
                logits, _, past_kv = model(x, use_cache=True)

            assert logits.shape == (BATCH, 8, VOCAB)
            for layer_kv in past_kv:
                k, v = layer_kv
                assert k.shape[0] == BATCH, f"k batch dim wrong: {k.shape}"
            ok(name)
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 18. MoE + KV cache — use_cache=True works with MoE layers
# ---------------------------------------------------------------------------
def test_moe_kv_cache():
    print("\n[18] MoE + KV cache — forward and decode correctness")
    for attn in ["mha", "gqa"]:
        name = f"moe+{attn}+rope"
        try:
            cfg   = tiny_config(attention_type=attn, use_moe=True)
            model = make_model(cfg)
            model.eval()

            prompt  = rand_ids(batch=1, seq=8)
            new_tok = rand_ids(batch=1, seq=1)
            full_in = torch.cat([prompt, new_tok], dim=1)

            with torch.no_grad():
                logits_full = model(full_in)[0][0, 8, :]

                out_prefill = model(prompt, use_cache=True)
                assert len(out_prefill) == 3, "MoE prefill must return 3-tuple"
                past_kv = out_prefill[2]

                out_cached    = model(new_tok, past_key_values=past_kv, use_cache=True)
                logits_cached = out_cached[0][0, 0, :]

            diff = (logits_full - logits_cached).abs().max().item()
            assert diff < 0.1, f"MoE KV cache mismatch: {diff:.4f}"
            ok(f"{name} (diff={diff:.5f})")
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 19. MLA + sliding window + KV cache
# ---------------------------------------------------------------------------
def test_mla_sliding_window_kv_cache():
    print("\n[19] MLA + sliding window + KV cache")
    SW = 6
    P  = 8
    for pos in ["rope", "yarn", "sinusoidal"]:
        name = f"mla+{pos}+sw{SW}"
        try:
            cfg   = tiny_config(attention_type="mla", positional_encoding=pos, sliding_window=SW)
            model = make_model(cfg)
            model.eval()

            prompt = rand_ids(batch=1, seq=P)
            with torch.no_grad():
                out_prefill = model(prompt, use_cache=True)
                assert len(out_prefill) == 3
                past_kv = out_prefill[2]

                for _ in range(4):
                    tok = rand_ids(batch=1, seq=1)
                    out = model(tok, past_key_values=past_kv, use_cache=True)
                    assert torch.isfinite(out[0]).all(), "non-finite logits"
                    past_kv = out[2]
                    for layer_kv in past_kv:
                        assert layer_kv[0].size(2) <= SW, \
                            f"KV len {layer_kv[0].size(2)} > window {SW}"

            ok(name)
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 20. LoRA — apply, forward, and only LoRA params get grads
# ---------------------------------------------------------------------------
def test_lora():
    print("\n[20] LoRA — apply + forward + only adapter params have gradients")
    try:
        from model.lora_utils import apply_lora_to_model
    except ImportError:
        fail("lora_import", "lora_utils not importable")
        return

    for preset in ["minimal", "attention_only"]:
        name = f"lora_{preset}"
        try:
            cfg   = tiny_config()
            model = make_model(cfg)

            lora_cfg = dict(
                use_lora=True,
                lora_preset=preset,
                lora_target_modules=[],
                lora_r=4,
                lora_alpha=8,
                lora_dropout=0.0,
            )
            model = apply_lora_to_model(model, cfg, lora_cfg)
            model = model.to(device=DEVICE, dtype=DTYPE)
            model.train()

            x       = rand_ids()
            targets = rand_ids()
            # PEFT wraps the model — call via model(x)
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out.logits
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
            loss.backward()

            # lora_B grads should be non-zero; lora_A grads are zero at init
            # (because lora_B=0 at init → chain-rule zeroes out lora_A grad)
            lora_grads = {n: p.grad for n, p in model.named_parameters()
                          if "lora_" in n and p.grad is not None}
            assert len(lora_grads) > 0, "No LoRA adapter gradients at all!"
            lora_b_grads = [g for n, g in lora_grads.items() if "lora_B" in n]
            assert lora_b_grads, "No lora_B gradients found!"
            assert any(g.norm().item() > 0 for g in lora_b_grads), \
                "All lora_B gradients are zero"
            ok(f"{name} ({len(lora_grads)} adapter grad tensors)")
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# 21. Mamba2 backward pass
# ---------------------------------------------------------------------------
def test_mamba2_backward():
    print("\n[21] Mamba2 backward — gradients flow")
    try:
        cfg = ModelConfig(
            model_architecture="mamba2",
            tokenizer_name="Qwen/Qwen2.5-0.5B",
            d_model=64, n_layers=2, vocab_size=VOCAB,
            max_seq_len=64, state_size=16, expand_factor=2,
            headdim=16, chunk_size=8,
        )
        model = make_model(cfg)
        model.train()

        x       = rand_ids(batch=1, seq=SEQ)
        targets = rand_ids(batch=1, seq=SEQ)
        out     = model(x)
        logits  = out[0]
        loss    = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        loss.backward()

        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert grad_norms, "no gradients in Mamba2"
        assert max(grad_norms) > 0, "all-zero Mamba2 gradients"
        ok(f"mamba2 backward (max_grad={max(grad_norms):.4f})")
        del model
    except Exception as e:
        fail("mamba2_backward", e)


# ---------------------------------------------------------------------------
# 22. Edge cases — single-token input, seq_len=1
# ---------------------------------------------------------------------------
def test_single_token_input():
    print("\n[22] Edge cases — single-token forward pass")
    for attn in ATTN_TYPES:
        name = f"{attn}+rope+seq1"
        try:
            cfg   = tiny_config(attention_type=attn)
            model = make_model(cfg)
            model.eval()

            x = rand_ids(batch=1, seq=1)
            with torch.no_grad():
                out = model(x)
            logits = out[0]
            assert logits.shape == (1, 1, VOCAB)
            assert torch.isfinite(logits).all()
            ok(name)
            del model
        except Exception as e:
            fail(name, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_all():
    global _PASS, _FAIL
    _PASS = _FAIL = 0

    print("=" * 65)
    print(f"LLM-Lab smoke tests  |  device={DEVICE}  dtype={DTYPE}")
    print("=" * 65)

    sections = [
        test_forward_all_combos,
        test_kv_cache_single_step,
        test_kv_cache_multistep,
        test_backward_all_attn,
        test_rlhf_tuple_unpacking,
        test_sft_forward,
        test_mamba2_forward,
        test_moe_forward,
        test_activation_functions,
        test_norm_types,
        test_sliding_window_kv_cache,
        test_alibi_decode_bias_shape,
        test_gradient_checkpointing,
        test_dpo_loss_function,
        test_sliding_window_mask,
        test_cache_backward_compat,
        test_kv_cache_batched_prefill,
        test_moe_kv_cache,
        test_mla_sliding_window_kv_cache,
        test_lora,
        test_mamba2_backward,
        test_single_token_input,
    ]

    section_fails = 0
    for fn in sections:
        before_fail = _FAIL
        fn()
        if _FAIL > before_fail:
            section_fails += 1

    print("\n" + "=" * 65)
    print(f"Individual tests : {_PASS} passed, {_FAIL} failed")
    print(f"Test sections    : {len(sections) - section_fails}/{len(sections)} clean")
    print("=" * 65)

    if _FAIL:
        sys.exit(1)
    else:
        print("All smoke tests passed!")


if __name__ == "__main__":
    run_all()
