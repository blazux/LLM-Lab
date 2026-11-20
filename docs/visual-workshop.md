# Model Assembly Workshop

Build a model by connecting real operations, then pretrain and export the graph
you assembled. The first version supports a dense causal decoder. Each available
layer changes the computation and is preserved in checkpoints and exports.

For a known starting architecture, see [editable Llama, Gemma and Qwen templates](architecture-templates.md).

## Start exploring

1. Run `bash gui/start.sh`, create a pretraining project, and open **Model**.
2. Choose **Explore a small Transformer**. The example uses `d_model = 64` and
   two decoder blocks. It is a starting point for learning, not a trained model.
3. Select **Tokenizer / Input IDs**, enter a local tokenizer directory or a
   Hugging Face ID, and click **Read tokenizer**. Vocabulary size affects the
   parameter count and is read again before training.
4. Read the diagram **from bottom to top**: Input IDs → Token Embedding →
   Decoder Block × N → RMSNorm → LM Head. Arrows show data flow.
5. Double-click the repeated block to inspect its layers. This example uses a
   modern **pre-norm** layout: RMSNorm → Causal Self-Attention → Residual Add →
   RMSNorm → Feed-Forward Network → Residual Add. The bypasses run around the
   sublayers. It is not an exact reproduction of the original post-norm
   Transformer in *Attention Is All You Need*.
6. Select a layer to read its explanation and equation. Change the number of
   repetitions and watch the parameter count change. Each repetition has its
   own weights.
7. Click **Test forward & backward**. This builds the model, computes a loss and
   checks gradients. Success means the computation runs; it does not measure
   the quality of a trained model.
8. Select your data and a short training run in the following steps. Use dataset
   name `text`, `json` or `parquet` with a local file path if desired. Observe
   the loss, then export your checkpoint.

You can also **Start from a blank canvas**: connect Input IDs → Token Embedding →
LM Head. This is a valid small model. Add attention to let it use context.
Drag from a layer’s **top output** to the next layer’s **bottom input**.
Residual Add receives its bypass through the **left input** (`b`).
**Arrange layers** lays out the current graph and its nested blocks upward;
existing diagrams using the old downward convention are migrated when opened.

## Learn by changing a residual connection

Residual Add takes two tensors of the same shape. In the example, `a` receives
the sublayer output and `b` receives the original residual stream. Delete a
connection and read the validation error to see which input is missing.
To try a model without a residual connection, remove the addition and connect
the sublayer directly to the following layer.

Compare runs with the same data and settings. Change one thing at a time.
**Undo** and **Redo** let you retrace your steps. Changing the architecture
requires a new run; a valid model can still learn poorly, which is a useful
experimental result.

## Names and diagram fidelity

The palette uses common terminology: **Token Embedding**, **Positional Encoding**,
**RMSNorm**, **LayerNorm**, **Causal Self-Attention**, **Feed-Forward Network
(FFN / MLP)**, **Residual Add**, **Decoder Block × N**, and **LM Head**.
Select a layer to configure it; settings belong to that specific layer, including
inside repeated blocks.

| Layer | Available configurations |
| --- | --- |
| FFN | SwiGLU, GeGLU, ReGLU, GELU, ReLU, ReLU², SiLU, Leaky ReLU; hidden dimension and linear biases |
| Self-Attention | MHA, MQA, GQA, simplified MLA with shared latent KV compression; head counts, latent dimension, optional Q/K RMSNorm and sigmoid output gate |
| Attention mask | Full causal or a sliding window, counted including the current token |
| Positions inside attention | RoPE with fraction/base settings, ALiBi distance bias, or None |
| Additive Positional Encoding | Fixed sinusoidal encoding with configurable base, or learned position embeddings |
| Normalization | RMSNorm or LayerNorm, with epsilon and dimension settings |

Gated FFNs use three learned projections; standard FFNs use two. Their parameter
counts, equations and outputs change with the selected activation.

### Where positional information belongs

For sinusoidal or learned absolute positions, add **Positional Encoding** after
Token Embedding and connect it into the data path. Set each attention layer’s
positional method to **None** for an absolute-position baseline. Combining
methods is allowed as an explicit experiment.

For **RoPE**, select it inside Self-Attention: it rotates queries and keys,
rather than adding vectors to embeddings. **ALiBi** similarly belongs inside
attention because it adds a distance-dependent bias to attention scores. The
selected method is visible on the attention node. **YaRN** is a context-extension
technique and is not presented as a base-model layer.

Attention is still a compound operation: its Q/K/V projections, optional Q/K
RMSNorm, positional method, attention and output projection cannot yet be wired
separately. The MLA option is a simplified shared low-rank KV bottleneck, not an
exact DeepSeek architecture or an optimized compressed KV cache. Sparse
attention, Per-Layer Embeddings and MoE require additional executable layers.
LM Head returns logits; Softmax converts them to probabilities when needed.

## What the graph means

- **Connections** define execution. Moving or arranging nodes only changes the
  presentation, including the positions stored in the project.
- Tensor dimensions and exact parameter counts come from the model runtime.
- Repeated blocks contain editable subgraphs with independent weights.
- Disconnected layers, cycles and unknown parameters prevent validation.
- The graph signature ignores node positions. Resuming a checkpoint with a
  different architecture is rejected rather than silently ignoring edits.

## Export and current limits

Exports contain weights, tokenizer, `assembly.json`, configuration and standalone
Transformers model code. Loading this custom format requires
`trust_remote_code=True`. Publishing to the Hub remains an explicit action in
the existing export flow; a local export does not publish anything.

The [experimental modern integration](modern-architectures.md) adds MoE, QSA, Gated DeltaNet, PLE and gated residual streams. Visual training workflows and dropout layers remain pending. Generation recomputes the prefix;
an optimized KV cache is still pending. The exported Transformers adapter
supports unpadded sequences. SFT and RL inherit their architecture from the
selected checkpoint instead of letting project settings replace it.

## Verification

```bash
LLMLAB_TEST_DEVICE=cuda python -m unittest discover -s tests -p test_visual_assembly.py -v
cd gui/frontend
npm test -- --maxWorkers=1
npm run build
```

Tests cover rewiring effects, dimensions, independent repetitions, exact resume,
and Transformers reload in an independent process. DOM tests cover editing,
terminology, variant settings, upward layout and port direction, using mocked network responses.
These do not replace a visual review in a browser.

For choosing a corpus and configuring held-out documents in the interface, see
[the dataset guide](datasets.md).

## Run metrics

The Run view plots training and validation loss, reported training and validation
perplexity, and learning rate. Throughput appears when supplied by the backend.
Perplexity has a logarithmic scale by default; turn it off for a linear view.
Latest training/validation perplexities include the step of each observation.
The frontend merges training and evaluation messages at matching steps, so one
series does not overwrite the other. Charts collect live events while the view
is open; full history is not restored after a reload.

The current trainer reports `exp(loss)`, with a numerical cap. Some model losses
include auxiliary objectives (MoE balancing and QSA indexer losses for Qwen4), so
this reported value is an approximation of language perplexity. The UI labels
it accordingly and never silently invents perplexity from an incoming loss.
Pure next-token perplexity requires exponentiating the token-averaged language
cross-entropy alone. The running training process is not restarted to change
that convention when updating this view. Compare like-for-like tokenizers,
validation data and metric definitions.
