# What do residual connections do?

Compare two small models that start with the same learned weights and see what
changes when their residual paths are removed. The exercise uses the supplied
synthetic corpus from [the first learning experiment](learning-path.md).

## What changes?

In the assembly workshop, a residual addition combines a transformation's output
with the representation that entered it. If that input is called `x` and the
transformation is `f`, the output is `x + f(x)`. Without the bypass and addition,
the output is just `f(x)`.

The **with-residuals** variant retains the attention and feed-forward bypasses
in each decoder block. The **without-residuals** variant removes those additions
and reconnects the computation in sequence. Both retain their normalization,
attention and feed-forward operations. Removing a bypass does not remove their
learned weight matrices.

The experiment changes executable graphs, not just a drawing. Inspect the
connections and read the diagram's legend before starting.

## Run the two variants

1. Install [LLM-Lab](installation.md) and open the residual-connections guided
   experiment from its dashboard, or append
   `#experiment=residual-connections` to your own Lab address.
2. Keep the default update budget and seed for your first pair. Choose an
   available NVIDIA GPU or CPU. The selected device is recorded with the run.
3. Run the variant with residual connections and wait for it to finish.
4. Start the variant without residual connections using the same settings.
   The two runs execute separately; opening the page or importing a recipe does
   not launch either one.
5. Select the recorded runs as A and B in the comparison view. Reopening the
   comparison preserves its selected run identities.

The [Understanding AI page](https://www.thelastserver.com/understanding-ai/#experiment-residual-connections)
also supplies an importable preset. It selects the experiment and its starting
settings; review the device and variant before launching.

## Check what is comparable

Read the comparison's configuration checks before interpreting the curves.
For this exercise, the corpus, tokenizer, seed, initial learned weights,
optimizer settings, update budget and compute environment should match. The
residual graph connections should differ. Missing evidence is not a verified
match; the view can still show two runs while flagging their differences.

Compare completed runs with the same amount of training. A stopped or failed
run only describes work that actually finished. Older runs may lack some of
the fingerprints or gradient observations recorded by newer versions.

## Read the measurements

- **Training and validation cross-entropy** measure next-token prediction error.
  Follow their evolution together; lower training error alone does not establish
  improvement on held-out examples.
- **Gradient norm before clipping** is the measured global size of the parameter
  gradients before their magnitude is limited. It is not the clipped norm and
  does not reveal every layer's gradient. A larger value alone is not evidence
  that a model is learning worse.
- **Fixed-prompt continuations** use the same prompts and generation settings.
  Compare their actual text alongside the validation scores, not as a substitute
  for evaluation.

Both variants may learn this shallow, controlled task. An advantage, a small
difference or no clear difference is an observation to explain, not a failed
exercise. One pair of runs does not establish a general ranking of architectures;
repeat with other seeds before drawing broader conclusions.

Read [the Transformer explanation](https://www.thelastserver.com/understanding-ai/#tech-transformer)
for the surrounding architecture. The
[Pre-LN explanation](https://www.thelastserver.com/understanding-ai/#tech-pre-layernorm)
discusses where normalization sits relative to transformations and residual
paths. This experiment keeps that placement fixed.
