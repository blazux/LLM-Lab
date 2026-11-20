# Editable architecture templates

Open **Model → Start from a known architecture**, choose an architecture and a
starting size, then click **Build editable template**. The same library is
available from **Architecture templates** while editing a graph. Applying it
replaces the current graph; **Undo** restores your previous model.

These templates build fresh models with random weights and retain your selected
tokenizer. They do not download weights, require access to a gated checkpoint,
or import a pretrained model. A family name identifies the starting architecture;
it does not promise the quality of a pretrained model or checkpoint compatibility.

The [modern architecture integration](modern-architectures.md) adds Gemma 4, Llama 4, Qwen 3.5 and Qwen4 preview. These new variants are experimental and await numerical/training validation.

## Initial families

| Starting point | Distinguishing choices |
| --- | --- |
| Llama 3 dense decoder | Pre-norm RMSNorm, GQA, SwiGLU, full standard RoPE with theta 500,000, unscaled token embeddings, untied LM Head |
| Gemma 1, 2B-style decoder | MQA, tanh-approximated GeGLU, unit-offset RMSNorm, embeddings scaled by sqrt(d_model), tied LM Head |
| Qwen 2.5, 0.5B-style decoder | GQA, SwiGLU, Q/K/V projection biases, full standard RoPE with theta 1,000,000, unscaled embeddings, tied LM Head |

The definitions are grounded in the reference implementations and configuration:
[Llama 3](https://github.com/meta-llama/llama3/blob/main/llama/model.py),
[Gemma](https://github.com/google/gemma_pytorch/blob/main/gemma/model.py),
[Gemma configurations](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py),
and [Qwen 2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json).
These are specific generations: the Gemma template is not Gemma 2, 3 or 4, and
Qwen 2.5 is not a Qwen3 hybrid architecture.

Standard RoPE uses inverse frequencies `theta ** (-2i / head_dim)` and the
standard rotation direction. Existing graphs retain the historical LLM-Lab
frequency convention unless explicitly changed, preserving old checkpoints.
The selected convention is exposed on each attention node.

## Choosing a size

Nano, Tiny, Small and Medium are **lab sizes**, not names of published checkpoints.
Each choice shows dimension, block count, query/KV heads, FFN size and context
length before applying it. Qwen sizes use dimensions divisible by its head
counts. Parameter count comes from the built graph and your tokenizer vocabulary.

Start small, run **Test forward & backward**, then try a short training run.
The probe has a 25-million-parameter limit; it is separate from training.

## Resize globally or customize individual layers

The behavior below describes the original three templates. Modern templates use per-layer subgraphs and different depth rules; see the [modern size-change contract](modern-architectures.md#size-changes).

Under **Your model → Resize the whole model**, set a global dimension, layer
count and context length, then click **Apply global size**. The runtime:

- Updates the existing graph instead of rebuilding the original template.
- Preserves connections, activations, normalization, positions, biases and gates.
- Scales each FFN's existing expansion ratio, rounding its hidden size to a
  multiple of eight, and scales MLA latent dimensions where present.
- Picks valid query/KV head counts, preserving head dimension and grouping as
  closely as the new width permits.
- Changes the count of the main top-level repeated block. Nested repetition
  counts are preserved. A custom graph with zero or multiple top-level repeated
  blocks needs individual block-count edits instead.

The parameter count and validation refresh after applying the change. The
result remains a graph you can edit: double-click **Decoder Block × N** and
configure each layer, reconnect sublayers, or add/remove nodes. **Undo** works
for both template selection and global resizing.

Resizing changes architecture, not existing weights. Start a fresh training run;
resume rejects a graph that differs from the checkpoint. The template origin
is kept in project configuration, checkpoints and export metadata, while the
actual modified graph remains authoritative.

## Verification

```bash
LLMLAB_TEST_DEVICE=cuda python -m unittest discover -s tests -p test_architecture_templates.py -v
```

The legacy suite compiles the original family/size combinations and compares their
logits with a Transformers reference model using identical weights, checks that
global resizing preserves user edits and gradients, and reloads every family
export in an independent process. DOM tests cover selection, resizing, editing
and undo. Browser visual review remains a separate check.
