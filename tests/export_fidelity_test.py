"""
Export fidelity test — verifies that a model exported to HuggingFace format
produces EXACTLY the same logits as the training-time model.

This is the guarantee behind "push to HuggingFace": if this test passes, the
uploaded model behaves identically to the one you trained.

Run: python tests/export_fidelity_test.py
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.config import ModelConfig
from model.factory import build_model
from export.hub import export_to_local

TOLERANCE = 1e-4

CASES = [
    ("gqa+rope+swiglu+tied", {"n_kv_heads": 2}),
    ("mha+rope+swiglu+untied", {"attention_type": "mha", "n_kv_heads": None,
                                "tie_word_embeddings": False}),
    ("mqa+rope+swiglu", {"attention_type": "mqa", "n_kv_heads": None}),
    ("mla+rope+swiglu", {"attention_type": "mla", "n_kv_heads": None}),
    ("gqa+moe", {"n_kv_heads": 2, "use_moe": True, "num_experts": 4,
                 "num_experts_per_token": 2}),
    ("gqa+gate+nope+partial-rope", {"n_kv_heads": 2, "use_attention_gate": True,
                                    "nope_every": 2, "rope_fraction": 0.25}),
    ("hybrid-gdn-1:1", {"n_kv_heads": 2, "hybrid_pattern": "1:1"}),
    ("gqa+sparse-dsa", {"n_kv_heads": 2, "use_sparse_attention": True, "sparse_top_k": 6}),
    ("gqa+moe-modern", {"n_kv_heads": 2, "use_moe": True, "num_experts": 4,
                        "num_experts_per_token": 2, "moe_router_type": "sigmoid",
                        "moe_aux_free_bias": True, "moe_shared_expert": True}),
]


def run_case(name, kwargs):
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                      vocab_size=256, max_seq_len=64, **kwargs)
    model = build_model(cfg).eval()

    tmp = tempfile.mkdtemp()
    ckpt = os.path.join(tmp, "best_model.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": cfg, "step": 1}, ckpt)
    out = os.path.join(tmp, "export_" + name.replace("+", "_"))
    with contextlib.redirect_stdout(io.StringIO()):
        export_to_local(ckpt, out, model_config=cfg)

    # Clear the transformers remote-code cache so each case reloads fresh code
    shutil.rmtree(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules"),
                  ignore_errors=True)
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True).eval()

    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        ref = model(x)[0]
        hf = hf_model(x).logits
    diff = (ref - hf).abs().max().item()
    shutil.rmtree(tmp, ignore_errors=True)
    return diff


def main():
    failed = 0
    for name, kwargs in CASES:
        try:
            diff = run_case(name, kwargs)
            ok = diff < TOLERANCE
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed += 1
            continue
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: max |logit diff| = {diff:.2e}")
        if not ok:
            failed += 1

    print()
    if failed:
        print(f"{failed}/{len(CASES)} export fidelity cases FAILED")
        sys.exit(1)
    print(f"All {len(CASES)} export fidelity cases passed — "
          "exported models reproduce training forward exactly")


if __name__ == "__main__":
    main()
