# 🧪 LLM-Lab

**1.0.0-rc.1 — release candidate.** Build, train and inspect small language models
on your own machine. Source code is available under the [MIT license](LICENSE).

<p align="center">
  <img src="assets/science.jpg" alt="LLM-Lab logo" width="200">
</p>

LLM-Lab combines an executable visual model workshop with guided experiments.
Connect layers, train from random weights, compare saved runs and inspect what
changed. The [capability guide](docs/capabilities.md) distinguishes the dense
pretraining workflow from experimental architecture and post-training options.

## Start the interface

Use Linux with Python 3.11–3.13, Node 20+, Git, curl and Python venv support.
CUDA additionally requires a compatible NVIDIA GPU and driver. The launcher
installs the constrained Python dependencies and the frontend dependencies.

```bash
git clone https://github.com/blazux/LLM-Lab.git
cd LLM-Lab
bash gui/start.sh
```

Open the GUI address printed by the launcher, normally
**http://127.0.0.1:5174**. Keep the terminal open. Device selection defaults to
CUDA when a working NVIDIA GPU is detected, otherwise CPU. Choose an explicit
installation device with `LLMLAB_DEVICE=cpu` or `LLMLAB_DEVICE=cuda`.

See [installation](docs/installation.md) for environment checks, ports, containers
and optional accelerators. The checked-out source and locally built images use
the current code; an older published image may not contain the same features.

## Learn by experimenting

The dashboard offers two guided paths.

### Learn: what a small model learns

Open [What does a small model learn?](docs/learning-path.md) to train on the
supplied synthetic corpus and compare held-out prediction errors and fixed-prompt
continuations before and after training.

### Learn: what residual connections do

Open [What do residual connections change?](docs/residual-connections.md) to train
the same learned layers with and without their bypass additions. Inspect the
validation curves, pre-clipping gradient norms and comparison controls.

Choose the device, budget and seed before starting. Opening a path or importing
its settings does not start training. The API runs one training job at a time;
results, configuration and logs remain available after reloading the page.

The paths link to the relevant explanations in
[Understanding AI](https://www.thelastserver.com/understanding-ai/). The site's
English and French explanations also provide settings files that can be imported
into the Lab. Illustrations on the site and measurements from a real run are
identified separately.

A short synthetic-corpus exercise demonstrates learning of repeated patterns.
It does not establish general language competence or guarantee that one
architecture will outperform another. For a larger corpus, see the
[TinyStories pretraining tutorial](docs/reference-pretraining.md).

## Build a custom model

Create a **Pretrain** project and choose an architecture template or a blank
canvas. Follow **Model → Dataset → Training → Run**:

1. Connect operations in the [visual workshop](docs/visual-workshop.md), inspect
   tensor dimensions and parameters, then test forward and backward computation.
2. Select [training and validation documents](docs/datasets.md). Read the
   tokenizer's actual vocabulary before training.
3. Set the optimizer-update budget, batch, accumulation and evaluation frequency.
4. Start a new run. Stop and resume saved checkpoints, inspect curves and compare
   generations using the checkpoint's own configuration.

Moving a layer changes the drawing; changing a connection changes the
computation. A valid graph can still learn poorly. Changing the graph or
vocabulary starts a new experiment rather than silently replacing a saved model.

The [graphical first-run guide](docs/first-run.md) explains the complete workflow.
The [architecture templates](docs/architecture-templates.md) are editable,
scaled educational models initialized with random weights. Modern hybrid and
MoE templates, SFT and policy optimization remain experimental; see
[architecture scope](docs/modern-architectures.md) before using them.

## Run a saved recipe

The noninteractive CLI can validate, train, stop, resume, generate and export a
model from a TOML recipe:

```bash
python src/cli.py validate recipes/small-model-learning.toml --check-data
python src/cli.py run recipes/small-model-learning.toml --output-dir outputs/my-first-model
python src/cli.py status outputs/my-first-model
python src/cli.py generate outputs/my-first-model --prompt "Ada finds"
```

Activate the installed Python environment first, or replace `python` with
`.venv/bin/python`. This supplied recipe uses CPU by default; edit its
`experiment.device` to `cuda` for a compatible CUDA runtime. The GUI offers its
own visible device selector. Use a new output directory for each new run.

See [recipes and CLI commands](docs/experiments.md) for exact resume, extensions,
artifact contents and export verification.

## Saved runs and exports

The API stores its registry in `data/runs.sqlite3` and its default outputs under
`data/runs/<run-id>/`. Set `LLMLAB_DATA_DIR` before startup to choose another
location. In Docker, persist `/app/data`. Back up the registry together with
its checkpoint and tokenizer folders. CLI recipes write to the directory
specified with `--output-dir`.

**Model Tools** loads recorded checkpoints for generation and export. A local
export contains the weights, tokenizer, configuration and standalone model code.
Custom exports require `trust_remote_code=True`; review that code before loading
an unfamiliar package. Publishing to the Hugging Face Hub is an explicit action,
with private repositories selected by default.

The source license does not replace dataset or model-license requirements.
Review the dataset license and choose the exported model's license before sharing.

## Documentation and checks

Start at the [documentation index](docs/index.md). The
[changelog](CHANGELOG.md) includes checkpoint and runtime migration guidance.

With the development dependencies installed in the same environment:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_test.py
python tests/export_fidelity_test.py
npm --prefix gui/frontend ci
npm --prefix gui/frontend test
npm --prefix gui/frontend run build
```

The fixtures use local data and tokenizers. CUDA checks use
`LLMLAB_TEST_DEVICE=cuda` and require an accessible compatible GPU. The
[capability guide](docs/capabilities.md) includes reproducible architecture and
profiling commands; timings depend on the device, shape, precision and protocol.
