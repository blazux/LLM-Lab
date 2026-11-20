# Changelog

## Unreleased

- Added two guided experiments: small-model learning and residual connections,
  with explicit CPU/CUDA selection, bounded training budgets and supplied data.
- Added before/after validation measurements and fixed-prompt generations, durable
  histories and honest interrupted/stopped states after reopening the interface.
- Added reusable A/B comparison with configuration differences and checks for
  data, tokenizer, initialization, training settings and execution environment.
- Recorded real pre-clipping gradient norms and before/after protocol fingerprints.
- Linked the guided paths to Understanding AI and added settings import from the
  English and French site without starting an experiment automatically.

## 1.0.0-rc.1 — 2026-10-19

Release candidate for the dense pretraining workflow. See the
[capability guide](docs/capabilities.md) for supported and experimental features.

- Executable graphical assemblies: connections determine the trained model;
  validation identifies incompatible ports, dimensions and disconnected layers.
- English workshop with standard Transformer terminology, bottom-to-top flow,
  editable architecture templates and global/local sizing.
- Guided pretraining sources, document ranges, held-out validation and token budgets.
- Durable run registry, replayable curves/logs, isolated run selection and safe
  checkpoint resume into a new folder. Multiple viewers keep independent cursors.
- Language perplexity excludes auxiliary objectives and z-loss; historical
  metrics retain their original definition.
- Checkpoint-based generation, local Hugging Face export and private Hub sharing.
- Fixed modern exported models' nonpersistent rotary and PLE scaling buffers,
  stale inference caches after checkpoint replacement, and static path traversal.
- Updated frontend dependencies, including Axios/UUID and Tailwind 4
  with the dedicated PostCSS plugin. Existing custom theme is retained.
- Reproducible CPU/CUDA setup, constrained dependencies, CI checks, source release
  manifest, container distribution and short architecture qualification/profiling.
- Dense pretraining is the stable candidate path. Modern text hybrids/MoE, SFT,
  policy optimization, full-model compilation and optional accelerator kernels
  remain experimental. No pretrained-quality or 1B throughput guarantee.

### Migration

Back up the run database and checkpoint folders before upgrading. Old CLI
checkpoints can be imported through Run; curves that were never persisted cannot
be reconstructed. New language CE cannot be compared directly with legacy total
objective perplexity. Exact optimizer/data resume requires a supported checkpoint
and the same recipe, corpus and execution environment.

Re-export modern checkpoints with this version to receive the deterministic-buffer
fix. Existing export folders vendor their own runtime and are not updated by
upgrading LLM-Lab. Exports require the declared Transformers version and
`trust_remote_code=True`. Assembly generation currently supports unpadded
sequences and recomputes prefixes. Changing tokenizer, graph or vocabulary starts
a new experiment; it is not an exact resume.
