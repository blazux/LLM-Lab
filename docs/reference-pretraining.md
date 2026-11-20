# Train a small model on TinyStories

This example is available as a [declarative recipe and CLI workflow](experiments.md),
including persistent state, graceful interruption, exact resume and verified export.

This recipe tests actual learning beyond the minimal lifecycle check. It trains
a dense decoder from scratch on a bounded English storytelling corpus and exports
it locally. It does not fine-tune an existing model or publish to the Hub.

## Reproduce on CUDA

Download the pinned public files with the Hugging Face CLI:

```bash
hf download roneneldan/TinyStories --repo-type dataset \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --include data/train-00000-of-00004-2d5a1467fff1081b.parquet \
  --include data/validation-00000-of-00001-869c898b519ad725.parquet \
  --include README.md --local-dir /tmp/llmlab-tinystories-source

python examples/reference_pretraining.py \
  --source-dir /tmp/llmlab-tinystories-source \
  --output-dir outputs/my-reference
```

Use a new output directory. The script requires CUDA and refuses CPU fallback.
The experiment itself runs offline once the parquet files and Python dependencies
are present. The dataset is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories),
whose dataset card identifies the license as CDLA Sharing 1.0.

For a quick integration check, add `--steps 10 --train-documents 512
--validation-documents 64 --eval-batches 4` to the Python command. This changes the
corpus, tokenizer and schedule; it is not comparable to the full reference run.

## Fixed recipe

| Component | Configuration |
| --- | --- |
| Architecture | 6 layers, width 384, 6 query heads / 2 KV heads, FFN width 1024 |
| Bricks | GQA, RoPE, RMSNorm, SwiGLU, tied input/output embeddings |
| Tokenizer | Byte-level BPE, 8,192 tokens, fitted only on selected training stories |
| Data | First 40,000 unique train documents; 512 unique validation documents from the official validation split |
| Separation | Whole stories; normalized duplicates removed within and across both selected splits |
| Sequence length | 256 packed token IDs; 255 next-token targets per sequence |
| Batch | 8 sequences × 2 accumulation batches per weight update |
| Budget | 1,000 updates; 4,080,000 next-token targets |
| Optimizer | AdamW, LR 0.001, betas (0.9, 0.95), weight decay 0.1, clipping 1.0 |
| Schedule | 50-update warmup, cosine decay |
| Precision | Float32 master weights/AdamW moments; bfloat16 autocast |
| Evaluation | Before training, every 100 updates and at the end; same first 32 validation batches |
| Other | Seed 42; dropout, compile, MoE, document masks and gradient checkpointing disabled |

The validation subset is fixed and disjoint from the selected training stories;
only its first 32 batches are scored. This is an engineering reference, not an
independent benchmark or a claim of competitive language quality.

## Artifacts and verification

The output directory includes:

- `manifest.json`: dataset revision requested, hashes of source and selected data,
  tokenizer hash, environment, seed, parameter count and source revision.
- `source.patch` and `reference_pretraining.py`: implementation snapshot, recording
  the implementation used for the run. Review local changes before sharing these files.
- `training_config.json`, `model_config.json`, selected JSONL data and tokenizer.
- `before.json`, `metrics.jsonl`, `results.json`: measurements and continuations
  for three fixed prompts before and after training, with fixed sampling seeds.
- `run/`: resumable best/latest/final checkpoints and the generated training report.
- `export/`: model weights, tokenizer, model card and self-contained custom model code.
- `export-verification.log`: a separate Python interpreter reloads the export
  without repository imports, compares logits and checks greedy continuation.

The exported model uses `trust_remote_code=True`. Its custom code matches this
project's architecture; a vanilla Llama export would change the forward pass.

For separate source datasets, `TrainingConfig.validation_datasets` supplies the
validation data. Both sources then use all their sequences, instead of applying
the historical 90/10 sequence holdout. The resume cursor also works in this mode.

## Read your results

Compare held-out language cross-entropy and perplexity before and after training,
then read all fixed-prompt continuations. A lower prediction error does not
establish factual reliability, reasoning ability or coherent long-form writing.
Keep the same validation subset, tokenizer and generation settings when comparing
runs. Changing the short-test options changes the experiment and its token budget.

The output records elapsed time and memory for this run. Read their measurement
scope before comparing hardware; corpus preparation, tokenizer fitting and export
verification are separate from the timed training loop.
