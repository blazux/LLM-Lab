# Run an experiment from a recipe

The noninteractive CLI can validate a model recipe, create a run, stop/resume it,
generate text and build a verified Hugging Face export. The first recipe reproduces
[the TinyStories learning reference](reference-pretraining.md). Python 3.11+ and
the project's Python dependencies are required.

## First run

If the pinned source data is not already present:

```bash
hf download roneneldan/TinyStories --repo-type dataset \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --include data/train-00000-of-00004-2d5a1467fff1081b.parquet \
  --include data/validation-00000-of-00001-869c898b519ad725.parquet \
  --include README.md --local-dir /tmp/llmlab-tinystories-source
```

Then, from the repository root:

```bash
python src/cli.py validate recipes/tinystories.toml --check-data
python src/cli.py run recipes/tinystories.toml --output-dir outputs/my-model
python src/cli.py status outputs/my-model
python src/cli.py generate outputs/my-model --prompt "Once upon a time"
```

`./llm-lab.sh` accepts the same commands when your active `python3` has the
project dependencies. With no arguments it still opens the original menu.

Validation checks unknown fields, types, architecture dimensions, file paths,
training settings and export constraints. It reports an estimated parameter
count and token budget without loading weights or requiring GPU access.
`--check-data` also scans the documents, checks their format and deduplicates them
in a temporary directory. It does not fit a tokenizer or guarantee training quality.

The example recipe requires CUDA and refuses silent CPU fallback. For a CPU
experiment, explicitly set `experiment.device = "cpu"` in your own recipe.

## Stop, resume and extend

Ctrl+C or SIGTERM finishes the current accumulation group, evaluates and saves a
resumable checkpoint. A planned stop can exercise the same path:

```bash
python src/cli.py run recipes/tinystories.toml \
  --output-dir outputs/paused-model --stop-after 200
python src/cli.py resume outputs/paused-model
python src/cli.py resume outputs/paused-model --additional-steps 200
```

`--stop-after` is an absolute optimizer-update number. It does not shorten the
original learning-rate schedule. `resume` uses the copied datasets, tokenizer
and saved training settings; the original recipe/source files are not needed.
An extension starts a fresh schedule over the additional updates. Earlier
verified exports are preserved; an extension writes `export-step-N/`.

Resume verifies the hashes of the saved recipe, data, tokenizer and initial
configurations. Edited/missing files cause an error instead of silently changing
the experiment. Concurrent writes to the same run are blocked by a process lock.
A new run requires a new directory; existing artifacts are not overwritten.

## Recipe structure

The checked-in [TinyStories recipe](../recipes/tinystories.toml) is an editable
example, not Python code. It contains:

| Table | Purpose |
| --- | --- |
| `experiment` | Name, seed, explicit CPU/CUDA device and CPU thread count |
| `data` | Local train/validation files, format and optional document limits |
| `tokenizer` | Fit byte-level BPE on training texts, or copy an existing local tokenizer |
| `model` | Architecture fields from `ModelConfig`; vocabulary/tokenizer are derived |
| `training` | Optimizer, batch, accumulation, schedule, evaluation and performance settings |
| `evaluation` | Fixed prompts and sampling settings for before/after continuations |
| `export` | Automatic export, model description and a user-selected license identifier |

Paths are relative to the recipe file, except absolute paths. Data formats are
`text` (one document per nonempty line), `jsonl` (objects with a string `text` field)
and `parquet` (a `text` column). Whole documents are deduplicated within/across the
selected sources after whitespace normalization. The tokenizer sees only the
prepared training documents. No remote code is needed for corpus preparation.

For a pretrained local tokenizer:

```toml
[tokenizer]
type = "existing"
path = "../my-tokenizer"
```

The initial runner supports pretraining. SFT and policy optimization remain in
the legacy entry points. Verified recipe exports currently require RoPE; other
position encodings can be explored with `export.enabled = false`. Further
architecture combinations still need their own correctness/learning validation.

`training.use_8bit_optimizer` defaults to `false`, so merely installing
bitsandbytes does not change an AdamW recipe. Set it explicitly to `true` for an
8-bit AdamW/Muon setup; this requires bitsandbytes and CUDA. Keep the same optimizer precision when comparing runs.

## What a run contains

```text
my-model/
  recipe.toml                 original recipe
  recipe.resolved.json        resolved, defaulted recipe
  manifest.json               file hashes, prepared data counts, package versions
  model_config.json
  training_config.json
  data/                       copied, deduplicated train/validation JSONL
  tokenizer/                  tokenizer fitted/copied for this run
  before.json                 untrained validation metrics and continuations
  metrics.jsonl               persisted training/evaluation observations
  state.json                  preparing/training/stopped/completed/failed state
  results.json                before/after metrics and continuations
  checkpoints/                best/latest/final checkpoints and training report
  export/                     verified standalone model package
```

`status` prints the saved state as JSON. Run/resume commands also print
training progress. Current snapshots store absolute paths: keep the run in its
original directory when resuming. Exact continuation assumes the same code,
packages and device; file hashes alone cannot establish numerical reproducibility
across environments.

## Export verification

Completed runs export automatically unless disabled in the recipe. A standalone
command can export any saved checkpoint to a new destination:

```bash
python src/cli.py export outputs/my-model --output-dir outputs/my-model-share
```

The export contains weights, tokenizer, configuration, model card and custom
model code. A fresh isolated Python interpreter reloads it offline, compares
logits and checks a greedy continuation. `verification.json` is written only if
these checks pass; errors leave a `verification.log` for inspection. The command
does not upload anything. Choose the model license in the recipe before sharing;
the example's `other` value is a placeholder, not a claim that the dataset license
automatically applies to trained weights.

## Check the recipe workflow

CLI subprocess tests cover validation errors, stopped/completed state, exact
resume, extension, manual/automatic export, generation, changed-data rejection,
source-file independence and SIGTERM. Run them on the local GPU with:

```bash
LLMLAB_TEST_DEVICE=cuda python -m unittest discover -s tests -p 'test_experiment_cli.py' -v
```
