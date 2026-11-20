# What does a small model learn?

Train a small next-token model from random weights, then compare what it predicts
before and after training. The supplied corpus contains controlled English
sentence patterns. This is an exercise in learning and measurement, not a
general language benchmark or an instruction-following assistant.

## Open the experiment

Follow the [installation guide](installation.md) and open LLM-Lab. Choose
**Open guided experiment** on the project dashboard. You can also append
`#experiment=small-model-learning` to your own Lab address. Opening the page
does not start training.

The [Understanding AI timeline](https://www.thelastserver.com/understanding-ai/#experiment-small-model-learning)
introduces the experiment and provides a small JSON recipe. Choose **Import
recipe** in the guided view to load its settings. Importing selects the bundled
experiment; it does not download code, load model weights or start a run.

## Choose the device and budget

Use **Compute device** to select CPU or an available NVIDIA GPU. CUDA is selected
by default when the backend runtime supports it. A GPU also requires a
CUDA-enabled Torch installation and a compatible driver; a CPU-only Python
environment cannot use an otherwise installed GPU. See [installation](installation.md).
A request for CUDA fails explicitly if CUDA is unavailable.

The default is 120 optimizer updates, with a permitted range of 20–300. The seed
starts at 42. Keep the seed, data, tokenizer and budget unchanged when repeating
an experiment for comparison. Exact numerical equality across devices or
different library versions is not guaranteed.

After application dependencies are installed, this experiment needs no network
download. It uses the supplied training and validation files and learns its
byte-level BPE tokenizer from the training texts only.

## Run and observe

Select **Start experiment**. The Lab prepares a run folder, fits its tokenizer,
measures the untrained model, trains it, and measures the resulting checkpoint.
The run records configuration, corpus hashes, seed, hardware, events,
checkpoints and generated samples.

The view shows:

- **Validation cross-entropy:** the average error on next-token predictions in
  held-out documents. Lower values mean better predictions on this evaluation.
- **Validation perplexity:** the exponential of language cross-entropy. Compare
  values only with the same tokenizer and evaluation data.
- **Training curves:** prediction error and learning rate over optimizer
  updates. Training and validation observations remain separate.
- **Before/after continuations:** the same prompts, tokenizer, sampling settings
  and seeds applied to the initial and trained model. Read the actual outputs;
  fluent text is not guaranteed by a short run.
- **Tokens and timing:** the amount of training performed and its observed
  duration. The reported training-loop throughput includes evaluation and
  checkpoint overhead; it is not a kernel benchmark.

Training documents and validation documents are distinct, but share the same
synthetic vocabulary and grammar. Better validation predictions demonstrate
learning on these patterns. They do not establish broad language understanding.

You can stop a run, reopen its saved observations, or start another run with a
different update budget. Closing the page does not stop the backend worker.
Restarting the backend interrupts it; the history must not claim completion.
An early stop may occur before a resumable checkpoint exists.

## Connect the observations to the ideas

- [BPE](https://www.thelastserver.com/understanding-ai/#tech-bpe): how the supplied
  text becomes the token units being predicted.
- [Transformer](https://www.thelastserver.com/understanding-ai/#tech-transformer):
  how preceding tokens enter the computation.
- [Adam](https://www.thelastserver.com/understanding-ai/#tech-adam): how prediction
  errors lead to parameter updates; this recipe uses AdamW.
- [Learning from text](https://www.thelastserver.com/understanding-ai/#tech-gpt1):
  how next-token prediction supplies a training objective. The exercise does not
  reproduce the original GPT model or its training scale.

In the assembly workshop, selecting a supported layer also opens links to the
matching explanation. Attention links follow its configured variant and
positional method.
