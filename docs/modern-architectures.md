# Modern architecture templates — experimental integration

The workshop now includes nine new **text-backbone** variants. Each builds a real
editable graph with random weights. They are scaled experimental architectures,
not copies of published checkpoints or claims of matching trained-model quality.
See the [capability guide](capabilities.md) for supported execution paths and
limitations. Scaled text backbones do not reproduce the authors' full models,
multimodal training or complete training recipes.

## Available variants

| Template | Layer structure and distinguishing operations |
| --- | --- |
| Qwen 3.5 dense | Three Gated DeltaNet layers then one gated full attention; dense SwiGLU; partial RoPE |
| Qwen 3.5 MoE | Same mixer schedule; routed experts and a gated shared expert |
| Llama 4 Scout | MoE in each block; RoPE/NoPE schedule; chunked attention; L2 Q/K normalization on RoPE layers |
| Llama 4 Maverick | Alternating dense and MoE FFNs; RoPE/NoPE; shared expert; long-position temperature tuning |
| Gemma 4 E2B | Four local layers then one global; PLE; shared K/V; wider FFNs in shared layers |
| Gemma 4 E4B | Five local layers then one global; PLE; shared K/V |
| Gemma 4 31B | Five local layers then one global; independent local/global head dimensions; global K=V; dense FFN |
| Gemma 4 26B-A4B | Local/global attention; separate dense and routed FFN branches, with their own normalizations |
| Qwen4 preview | Qwen3.8-Flash-Next text architecture: GDN/QSA, four gated residual streams, n-gram injection in the second layer, routed/shared experts |

Nano/Tiny/Small/Medium reduce dimensions, layer counts, expert counts and n-gram
buckets. The schema allows configuring expert counts up to 512 and the published
top-k counts, but increasing them consumes memory even when few experts are
active per token. These are lab size presets, not published parameter sizes.

## What is visible and editable

Each layer is a block that can be opened. The graph explicitly connects:

- Hidden states between operations and layers.
- Token IDs to Gemma PLE and Qwen n-gram tables.
- Packed PLE features to each Gemma layer and its gated injection.
- K/V states from their producing Gemma layer to consuming layers.
- Residual context and write gates between Qwen4 Gated Residual Read/Write.
- Parallel dense and expert branches in Gemma MoE.

An edge selects an output with `sourceHandle` and an input with `targetHandle`.
The v2 validator checks port types and dimensions, cycles and disconnected nodes.
The graph runtime executes these connections; editing a subgraph edits the
actual model. Removing an auxiliary boundary port also requires updating its
parent connections. Existing v1 graphs remain supported.

Head dimensions are independent of model width in modern attention. DeltaNet
has independent Q/K and value head counts/dimensions and a causal depthwise
convolution. RoPE uses the reference model's partial/proportional convention.

## Size changes

Width-only changes preserve the whole graph and scale its dimensional fields.
Head counts, expert counts, top-k, windows, indexer budget and positional bases
stay explicit. Hidden head dimensions are rounded to valid multiples of eight.

Depth changes **rebuild the root graph**, retain surviving layer bodies, and add
new layers from the selected family schedule. Custom root-level additions may
be replaced; Undo restores the previous graph. Depth must be a multiple of the
family cycle. PLE/KV-sharing variants require two cycles so both local and global
K/V sources exist. Sharing thresholds are aligned to complete cycles using the
reference variants' unshared/shared proportions.

A resized model starts a new run. Resume continues to reject a different graph.

## Training and implementation scope

The complex operations adapt the **installed Transformers 5.20.1** reference
modules. The version is pinned in `requirements.txt`, and standalone exports
include the adapters, graph schema and a requirements file. No source code or
model weights are downloaded by constructing a template. Update the runtime
environment before using these templates:

```bash
python -m pip install -r requirements.txt
```

The trainer receives MoE balancing losses. QSA uses an explicit configurable KL
indexer objective against detached dense attention, because hard top-k selection
alone supplies no gradient to the selector. This is an experimental training
choice, not a reproduction of Qwen's complete training recipe. Set `index_loss`
on each QSA brick to control it. Set `aux_weight` on each MoE brick to control
its balancing objective.

Packed documents are isolated by attention masks, and recurrent/convolutional
modules restart at document boundaries. The n-gram EOS ID is an explicit node
setting; set it to the tokenizer's EOS ID when using EOS-segmented data.

Reference PyTorch kernels are functional fallbacks, not a claim of optimized
training throughput. The current runtime recomputes the prefix during generation;
it does not implement an optimized hybrid inference cache or asynchronous n-gram
offload. All weights normally reside on the selected device.

The templates cover the language backbones. Vision/audio encoders, multimodal
preprocessing and MTP auxiliary training heads are outside this integration.
The current assembly adapter supports unpadded input. SFT/RL remain experimental. Context settings do not guarantee long-context recall
or a particular inference speed.

## Inspect and check an architecture

The following script uses
random tokens, no downloaded dataset or tokenizer:

```bash
python examples/modern_architectures.py --list
python examples/modern_architectures.py --family qwen35_dense --device cuda
python examples/modern_architectures.py --family qwen4_preview --device cuda --length 32 --backward
python examples/modern_architectures.py --all --device cuda --backward --checkpoint --export
```

`--export` writes temporary checkpoints, reloads the standalone exported model
and compares logits. Random-token loss only checks execution. Use the interface
with your tokenizer and real data to assess learning, memory usage and usability.
The built-in quick probe retains its 25M-parameter limit.

## References

- [Qwen3.5-4B configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5 MoE configuration](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)
- [Llama 4 implementation](https://github.com/huggingface/transformers/tree/v5.20.1/src/transformers/models/llama4)
- [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json), [E4B](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json), [31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json), [MoE](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json)
- [Qwen3.8-Flash-Next / Qwen4 preview](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Pinned reference implementation](https://github.com/huggingface/transformers/tree/v5.20.1/src/transformers/models)
