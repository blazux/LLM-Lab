# Choose data for a first pretraining run

Dataset cards and GUI presets checked on **2026-10-16**. These are useful
starting points, not a universal ranking of data quality. The goal, language,
model capacity, tokenizer and available compute determine what works well.

## Which corpus?

| Goal | Corpus | Why choose it? | Main limitation |
| --- | --- | --- | --- |
| First small-model experiment | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | Short synthetic English stories with simple vocabulary; useful for observing language learning at small scale. | Narrow language and topics; it does not teach a general assistant. |
| Broader English pretraining | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Educational-quality selection of web pages. The GUI selects `sample-10BT`, rather than the full roughly 1.3T-token corpus. | Classifier scores are imperfect; broader capabilities take substantially more training. |
| French pretraining | [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Filtered, deduplicated multilingual web data; `fra_Latn` selects French. | General web quality varies. Check tokenizer efficiency on French. |
| Synthetic educational material | [SmolLM-Corpus / Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | About 39M synthetic textbooks, stories and articles; useful for studying an educational mixture. | Synthetic factual errors and repeated styles remain. |
| Mathematical specialization | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | About 14.7B tokens of filtered mathematical web text, retaining mathematical notation. | Not verified solutions and not a general language curriculum. |

Other established open pretraining references include
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
[DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0),
and [Dolma](https://huggingface.co/datasets/allenai/dolma).
FineWeb provides broad filtered English web text; DCLM studies data filtering
at scale; Dolma is a diverse corpus associated with the OLMo research program.
They are useful comparison points when designing a larger recipe. Corpus size
alone does not establish superiority; filtering, deduplication, domain coverage
and evaluations matter. See the [FineWeb research paper](https://arxiv.org/abs/2406.17557)
for educational filtering experiments.

## Use the interface

1. Create a **Pretraining** project and choose a small model. A dense model is
   easier to debug first; the new modern architecture integrations remain
   experimental (see [their guide](modern-architectures.md)).
2. Open **Training Datasets**. Select a learning goal, read the corpus description
   and limitations, then click **Use … — replace data sources**. This replaces
   the training and validation source lists; selecting a goal alone changes
   only the explanation. Source options remain editable.
3. Keep **Start with 40,000 training documents** checked for an initial test.
   The preset also configures 512 validation documents. This limit counts source
   rows, including empty rows that tokenization later skips.
4. Click **Apply short test training settings** if desired: 1,000 optimizer
   updates, 100 warmup updates, batch size 1, accumulation 4, evaluation every
   100 updates using up to 20 batches. These settings do not resize the model
   or alter your tokenizer, optimizer or output directory.
5. Select a tokenizer appropriate to the corpus. For the first English experiment,
   a small vocabulary (for example 8,192 tokens) avoids an oversized embedding
   and output layer. There is already a local tokenizer from the
   [reference experiment](reference-pretraining.md) if that run has been prepared.
   Use a new output directory for each experiment.
6. Check the token budget and memory estimate in **Training**, then launch from
   **Run**. Generate continuations before and after training using the same
   prompts. Keep the held-out source fixed when comparing architectures.

The short preset is a machinery/learning experiment, not a promise of fluent
text after 1,000 updates. For a context length of 512, its nominal token budget
is `1000 × 1 × 4 × (512 − 1) = 2,044,000` predicted tokens. A larger context
increases both the budget and resource use. A document limit and an optimizer
update limit are separate: the trainer repeats a finite stream until it reaches
Max steps. Repeated passes can encourage memorization.

Streaming still transfers data and can cache files. Reading 40,000 rows need
not transfer exactly those rows: storage formats and shards determine the
actual network traffic. Published token counts also use the publisher's
particular tokenizer; your tokenizer may produce a different count.

## What validation means

- TinyStories uses its native `validation` split, capped at 512 documents.
- Other guided presets reserve the first 512 rows of the selected training
  source for validation and start training at row 512. The published test split,
  where available, stays untouched. This is a reproducible initial diagnostic,
  not a random or representative benchmark sample.
- Presets pin immutable Hub commits. Changing a dataset revision, source files,
  ranges or order changes the experiment and invalidates assumptions of an
  exact checkpoint resume.
- The loader rejects overlapping training/validation row ranges of the same
  explicitly named source and split. This does **not** detect duplicate text
  across rows, splits, file aliases or corpora. The local reference preparation
  additionally removes normalized duplicates; use it when you want that stronger
  first-run baseline.
- Custom configurations without a separate validation source retain the legacy
  90/10 partition of packed sequences. A long document can contribute to both
  sets. The UI identifies this weaker fallback.

If training loss falls while validation loss stalls or rises, investigate
memorization, too little unique data, and train/validation distribution mismatch.
Compare loss/perplexity only with the same tokenizer, corpus and evaluation
procedure; a lower number on TinyStories does not prove a better general model.

## Custom data and mixtures

Each pretraining source supports `name`, `subset`, `split`, `revision`,
`text_field`, `data_files`, `skip`, `take`, and `weight`. Ranges apply before
text extraction/tokenization. Documents are encoded and EOS-terminated, then
packed into sequences. Nonempty text must be a string. A missing source or text
column fails explicitly rather than silently changing the mixture.

```json
{
  "datasets": [{
    "name": "HuggingFaceFW/fineweb-edu",
    "subset": "sample-10BT",
    "split": "train",
    "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "skip": 512,
    "take": 40000,
    "text_field": "text"
  }],
  "validation_datasets": [{
    "name": "HuggingFaceFW/fineweb-edu",
    "subset": "sample-10BT",
    "split": "train",
    "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "take": 512,
    "text_field": "text"
  }]
}
```

For local JSONL use `name: "json"`, `data_files: "/absolute/path/corpus.jsonl"`
and a text field per row. Builders `text` and `parquet` are also supported.
Plain text has one document per line. These new range/column controls apply to
**pretraining**; SFT and preference loaders have their own schemas.

With multiple sources, positive finite weights specify document sampling
probabilities, not token percentages. The current mixture stops when one source
is exhausted, and the trainer starts another pass. Sources with different
lengths or cardinalities can therefore contribute different token proportions
than the weights suggest. Start with one source before tuning mixtures.

Pretraining corpora are not conversation datasets. SFT uses demonstrations
(such as messages); preference training uses preference judgments or prompts,
depending on the algorithm. Those stages cannot replace learning language
from scratch.

Before sharing a result, keep dataset IDs, commits, configurations, ranges,
licenses, tokenizer and training settings with the model card. The UI links
the source cards and shows the published dataset license; underlying content
terms can still apply. The training configuration is saved with the run.

## Extend a capped pretraining run

An explicit `additional_steps` extension may increase (or remove) `take` for a
single training source. Its dataset ID, revision, subset, split, skip, text
column, files and weight must stay identical; validation also stays unchanged.
The checkpoint's weights, optimizer and current-pass data cursor are retained,
and a new warmup/decay phase begins at the checkpoint step. Earlier completed
passes cannot be undone. This is a continued experiment with a larger available
corpus, not an exact reproduction of the original capped training plan.
Changing source identity/order, shrinking the cap or changing mixtures still
requires a separate training setup. A normal resume without `additional_steps`
continues to require exactly the saved data configuration.
