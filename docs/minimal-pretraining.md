# Minimal pretraining lifecycle

This example covers a small dense Transformer: local tokenizer
and corpus, pretraining, graceful stop, checkpoint resume, generation, local
Hugging Face export, and reload outside the LLM-Lab source tree. It does not
publish anything to the Hub.

## Run the demonstration

With the project's Python dependencies installed, from the repository root:

```bash
python examples/minimal_pretraining.py --output-dir /tmp/llm-lab-demo
```

Choose a new output directory. The script creates its own tiny WordLevel
tokenizer and text corpus, stops after two weight updates, resumes to four,
then exports and reloads the result. CUDA is used automatically when available;
otherwise it runs on CPU. This verifies the machinery, not language quality.
The generated `export/` contains weights, tokenizer, configuration, model card
and the custom model implementation. Reload it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/tmp/llm-lab-demo/export"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, local_files_only=True,
)
```

## Regression checks

```bash
python -m unittest discover -s tests -p 'test_pretraining_lifecycle.py' -v
LLMLAB_TEST_DEVICE=cuda python -m unittest discover -s tests -p 'test_pretraining_lifecycle.py' -v
python tests/smoke_test.py
```

The lifecycle suite defaults to CPU. Its CUDA mode fails explicitly if no GPU
is accessible. It verifies that interrupted training matches uninterrupted
training, including dropout, optimizer/scheduler state and the data cursor. It
also checks accumulation against a larger batch in float32, extended training,
empty streams, added tokenizer tokens, and reload/generation in a separate
process. The fixtures require no model or dataset downloads. The optional PDF
report is omitted in the regression suite and exercised by the demonstration.

Weights and optimizer moments remain float32. For a first actual learning run
with a separate validation corpus, see [the reference recipe](reference-pretraining.md).

## Pretraining checkpoint semantics

- `max_steps`, `warmup_steps` and `eval_every` now count **optimizer updates**.
  One update processes `gradient_accumulation_steps` batches. Approximate token
  budget: `max_steps * batch_size * gradient_accumulation_steps * (max_seq_len - 1)`.
- Graceful stop finishes the current accumulation group before saving. Saved
  `step` is the number of completed weight updates.
- Version 2 checkpoints store optimizer/scheduler state, PyTorch CPU/CUDA RNG
  state and a cursor within the current data pass. Exact resume assumes the
  same recipe, data content/order and execution environment.
- Resume with the saved training configuration and `load_optimizer_state=True`.
  Set `additional_steps` to extend the run with a new schedule; the new target
  is persisted for subsequent resume.
- Old checkpoints used ambiguous batch counters and could omit accumulated
  gradients. Their weights remain usable with `load_optimizer_state=False`;
  this starts a new run at step zero, rather than claiming an exact resume.
- These checkpoint semantics apply to dense pretraining. SFT and policy optimization remain experimental.

For local data, dataset configuration accepts e.g.
`{"name": "text", "data_files": "/absolute/path/corpus.txt"}`. The existing
sequence-based validation split needs a sufficiently long corpus: at least
91 complete sequences for the default 90/10 partition to yield validation data.

For choosing a corpus and configuring held-out documents in the interface, see
[the dataset guide](datasets.md).

## Recorded runs in the interface

The **Run** step lists persisted runs for the current project. Selecting one
restores its saved configuration, progress, logs and metric curves. Editing the
project's model or training settings does not change a recorded run. **Start new
training from project settings** creates a separate run.

The local API writes its registry and replayable events to
`data/runs.sqlite3`, with default outputs under `data/runs/<run-id>/`. Set
`LLMLAB_DATA_DIR` before starting the backend to choose another registry root.
In Docker the default is `/app/data`; keep that directory on a persistent volume.
Custom output folders are registered too. Back up the registry together with the
output folders: the database contains the history, while checkpoints contain the
weights and optimizer/data state. Existing checkpoint folders are not overwritten
by a new API run.

- Reloading the page, changing views or reconnecting the event stream replays
  the saved events. Multiple viewers can monitor the same run independently.
- After a backend restart, an unfinished run is marked **interrupted**. It does
  not resume automatically. Choose **Resume saved checkpoint into a new run
  folder** to continue from the latest available saved checkpoint.
- Additional optimizer steps set a new target relative to the checkpoint step.
  With zero additional steps, resume continues its original saved budget. A
  completed budget requires a positive extension. Unsaved updates after the last
  checkpoint must be repeated.
- **Import an existing pretraining run** accepts a folder containing a supported
  checkpoint, including a `checkpoints/` subfolder. It restores saved configuration
  and final evaluation; live curves that were never recorded cannot be recovered.
- In **Model Tools**, select a recorded run to restrict Playground and Export
  checkpoint choices to that run, including custom output folders. The selected
  checkpoint supplies the model configuration and tokenizer.

### Reading the metrics

For new pretraining runs, `language_cross_entropy_v2` distinguishes:

- **Training objective**: the actual optimization loss, including auxiliary
  terms such as MoE balancing and z-loss.
- **Training / validation cross-entropy**: next-token language loss, averaged
  over tokens. Validation loss and best-checkpoint selection use this value.
- **Auxiliary objectives**: the remaining optimization terms, shown separately.
- **Perplexity**: `exp(language cross-entropy)`, excluding auxiliary objectives.
  Compare values only with the same tokenizer and evaluation data. Overflowing
  or unavailable values are omitted, not capped or manufactured. MAXIS sampled
  training loss does not provide full language perplexity; full validation does.

Imported older checkpoints retain their legacy metric label. Their perplexity
may include auxiliary losses and is shown with a warning. Resuming an old run
creates a new run using the current metric definition and resets the old best-loss
comparison, while preserving the original run's metrics. SFT and policy
optimization still use their existing metric definitions and are not covered by
this pretraining metric guarantee.
