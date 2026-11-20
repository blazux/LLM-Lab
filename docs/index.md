# LLM-Lab documentation

Start with [installation](installation.md), then choose a guided experiment or
build a custom model. The [capability guide](capabilities.md) describes the current
release candidate and the scope of experimental features. Read the
[changelog](../CHANGELOG.md) before upgrading an existing environment or checkpoint.

## Guided experiments

- [What does a small model learn?](learning-path.md): train on supplied text and
  compare validation errors and continuations before and after training.
- [What do residual connections change?](residual-connections.md): compare two
  variants with matching learned initialization, data and training settings.

Both paths connect observations to explanations in
[Understanding AI](https://www.thelastserver.com/understanding-ai/). The GUI keeps
measured results and identifies missing comparison information.

## Build and train a model

- [Your first graphical experiment](first-run.md): from model selection to a saved
  checkpoint, generation and local export.
- [Visual assembly workshop](visual-workshop.md): layers, connections, dimensions,
  forward/backward checks and graph editing.
- [Architecture templates](architecture-templates.md): dense families, size presets
  and explicit global/local changes.
- [Choose a dataset](datasets.md): tokenizer vocabulary, document ranges and held-out
  validation.
- [Modern architecture templates](modern-architectures.md): experimental hybrid,
  recurrent, sparse and MoE text backbones.

## Recipes, persistence and exports

- [Recipe CLI](experiments.md): validate, train, stop, resume, extend and export a
  saved experiment.
- [Minimal pretraining lifecycle](minimal-pretraining.md): an offline example with
  tiny local data, checkpoint semantics and recorded GUI runs.
- [Train on TinyStories](reference-pretraining.md): a pinned public corpus, a larger
  recipe and before/after evaluation.

## Advanced training reference

The [complete training guide](full-guide.md) covers model configuration, SFT,
LoRA and policy-optimization entry points. SFT and RLHF remain experimental;
read the capability guide before choosing them for a project.

For reproducible environment and container commands, use the installation guide.
Report errors with the version, device, relevant settings and a short reproduction;
remove credentials and private dataset content from shared logs.
