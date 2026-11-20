# Capabilities and limitations

LLM-Lab's core learning path is **small dense causal text pretraining**. Graph
connections define executable computation; runs keep their configuration,
measurements and checkpoints. Modern templates, SFT and policy optimization
remain experimental. The current version is a release candidate.

## Workflows

| Feature | Scope |
| --- | --- |
| Dense visual graph, raw-text pretraining, checkpoint stop/resume | Core learning workflow |
| Durable run history, language CE/perplexity, checkpoint generation/export tools | Core learning workflow |
| Guided small-model and residual-connection experiments | Supplied data, bounded budgets and saved before/after measurements |
| A/B comparison | Checks data, tokenizer, initialization, budgets, training settings and runtime; missing measurements remain unknown |
| Local export and optional private Hub sharing | Explicit user actions; custom exports include executable model code |
| Modern text backbones, MoE, recurrent/sparse attention | Experimental |
| SFT / PPO / DPO / GRPO, MAXIS, full-model compilation | Experimental |
| Visual training-workflow graph, multimodal training | Not provided by the current workshop |

Small synthetic exercises demonstrate computation and learning of patterns.
They do not establish pretrained-model quality or performance at billion-parameter
scale. A graph that computes and differentiates can still learn poorly.

## Architecture templates

Templates initialize random weights and scale dimensions for local exploration.
They do not download the corresponding pretrained model or reproduce its complete
training recipe. The [template guide](architecture-templates.md) explains sizing;
[modern architectures](modern-architectures.md) describes the experimental layers.

| Templates | Architectural scope |
| --- | --- |
| `llama3`, `gemma1`, `qwen25` | Scaled dense causal decoders |
| `qwen35_dense`, `qwen35_moe` | Experimental hybrid text backbones |
| `llama4_scout`, `llama4_maverick` | Experimental text backbones with routed FFNs |
| `gemma4_e2b`, `gemma4_e4b`, `gemma4_31b`, `gemma4_moe` | Experimental text backbones with explicit local/global and per-layer connections |
| `qwen4_preview` | Experimental Qwen3.8-Flash-Next text architecture |

Modern bricks adapt the pinned Transformers **5.20.1** implementations. Their
configuration and model modules are installed dependencies; constructing a
template does not fetch remote executable code. The relevant primary references
include:

- [Gemma 4](https://huggingface.co/docs/transformers/model_doc/gemma4): local/global
  attention and per-layer embeddings.
- [Llama 4](https://huggingface.co/docs/transformers/model_doc/llama4): text backbones,
  expert routing and RoPE/NoPE conventions.
- [Qwen 3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5): hybrid
  DeltaNet/attention layers and routed/shared FFNs.
- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next): the source of
  the template explicitly named Qwen4 **preview**.

Image/audio towers, MTP training heads and full published model sizes are outside
these templates. The QSA indexer KL auxiliary objective is an experimental Lab
training option; it does not reproduce the authors' complete training objective.

## Execution and measurement

Dense Self-Attention offers **Auto** (PyTorch SDPA dispatch) and **Math reference**.
Auto may choose a fused kernel only for compatible shapes, precision and masks.
The profiling tools record the operator actually dispatched. A setting alone does
not guarantee a particular kernel or speedup.

Modern reference attention uses eager execution. Reference recurrent and causal
convolution fallbacks remain available without optional accelerators, with
substantial performance limits. Optional FLA and causal-convolution packages are
not installed as part of the default environment. Compilation is optional.

Assembled graph generation currently recomputes the prefix; it does not implement
an optimized hybrid KV cache or asynchronous n-gram offload. Weights normally
reside on the selected device. Assembly exports support unpadded sequences and
require their declared dependency versions and `trust_remote_code=True`.

The trainer separates token-weighted language cross-entropy from auxiliary
objectives. Language perplexity is its exponential. Compare metrics only with
the same tokenizer, held-out data and metric definition. Historical runs can
retain an older, approximate definition; see the [migration notes](../CHANGELOG.md).

## Run your own checks

```bash
python scripts/qualify_architectures.py --device cpu --output /tmp/qualification-cpu.json
python scripts/qualify_architectures.py --device cuda --output /tmp/qualification-gpu.json
python scripts/profile_training.py --family llama3 --output /tmp/profile-dense.json
python scripts/profile_training.py --family qwen4_preview --output /tmp/profile-hybrid.json
python scripts/profile_training.py --family llama3 --compile-ffn --output /tmp/profile-compiled.json
```

Qualification checks construction, differentiation, optimizer replay and export
reload using short synthetic inputs. Profiling compares microbatches at a fixed
effective batch, performs warmup, synchronizes timed GPU work and records peak
allocation. Its timed region excludes data loading, validation and checkpoint I/O.
These results are different from the guided interface's mean training-loop rate,
which includes validation and checkpoint overhead. Keep hardware, precision,
shape and measurement scope alongside any results you compare.
