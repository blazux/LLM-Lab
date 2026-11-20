/**
 * Educational content for the model builder.
 *
 * LLM-Lab is a tool for learning how LLMs work by building one — every
 * component of the diagram gets a real explanation: what it does, why it
 * exists, what the trade-offs are, and who uses it in production.
 */

export interface GlossaryEntry {
  title: string
  /** What this component does and why it exists — 3-6 sentences, plain language. */
  body: string
  /** Practical trade-offs / what happens if you change it. */
  tradeoffs?: string
  /** Production models using this technique. */
  usedBy?: string
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
  tokenizer: {
    title: 'Tokenizer',
    body:
      'Neural networks work on numbers, not text. The tokenizer chops text into “tokens” — frequent words and word-pieces — and maps each to an integer ID. “unbelievable” might become [un, believ, able]. This is BPE (byte-pair encoding): the vocabulary is learned by repeatedly merging the most frequent character pairs in a corpus. Reusing a pretrained tokenizer (Qwen, LLaMA…) is standard practice: training your own adds little and breaks compatibility with existing ecosystems.',
    tradeoffs:
      'Bigger vocabulary → shorter sequences (cheaper attention) but a bigger embedding table. With a 151k vocab and a small d_model, the embedding table can dominate your parameter count — watch the live estimate when you shrink the model.',
    usedBy: 'Every LLM. Qwen2.5 (151k vocab), LLaMA (32k), GPT-4 (~100k).',
  },
  embed: {
    title: 'Token Embedding',
    body:
      'A lookup table turning each token ID into a learnable vector of size d_model. These vectors are where the model stores what a token “means” before any context is applied — related tokens end up with nearby vectors during training. This lab also multiplies embeddings by √d_model at input (an original-Transformer detail that balances their scale against the positional signal).',
    tradeoffs:
      'Table size = vocab_size × d_model parameters. With tie_word_embeddings the same table is reused as the output projection, halving that cost — standard for small models.',
    usedBy: 'Universal. Weight tying: GPT-2, Gemma, Qwen3 ≤ 4B.',
  },
  pos: {
    title: 'Positional Encoding',
    body:
      'Attention is a set operation: without extra information, “dog bites man” and “man bites dog” look identical. Positional encodings inject token order. RoPE (Rotary Position Embedding) is today’s standard: it rotates each query/key vector by an angle proportional to its position, so attention scores depend only on the RELATIVE distance between tokens — which generalizes better to unseen lengths than absolute positions. YaRN stretches RoPE for longer contexts; ALiBi skips rotation and simply penalizes attention by distance.',
    tradeoffs:
      'rope_fraction rotates only part of each head (partial RoPE, Qwen3-Next uses 25%) leaving the rest position-free. nope_every removes positional encoding entirely on every Nth layer (iRoPE) — those layers see pure content, which helps long-context reasoning.',
    usedBy: 'RoPE: LLaMA, Qwen, DeepSeek. Partial RoPE + NoPE layers: Qwen3-Next, Llama 4, SmolLM3.',
  },
  norm: {
    title: 'Normalization',
    body:
      'Deep networks are numerically fragile: activations can drift, explode, or collapse as they pass through dozens of layers. A normalization layer rescales each token vector to a stable magnitude before every sub-layer (this is “pre-norm”, the layout that made deep Transformers trainable without tricks). RMSNorm is LayerNorm minus the mean-centering — slightly cheaper, equally effective, and the modern default.',
    tradeoffs:
      'norm_eps only matters for numerical safety — leave it. The interesting knob you don’t see here: this lab also applies QK-norm inside GQA attention, normalizing queries/keys per head, which lets you train at higher learning rates without logit explosions.',
    usedBy: 'RMSNorm: LLaMA, Qwen, DeepSeek, Gemma. QK-norm: Qwen3, OLMo 2, Gemma 3.',
  },
  attn: {
    title: 'Attention',
    body:
      'The heart of the Transformer. Each token builds a Query (“what am I looking for?”), and every token offers a Key (“what do I contain?”) and a Value (“what do I pass along if selected”). A token’s output is the Value-average of all previous tokens, weighted by how well its Query matches their Keys. Multiple heads run this in parallel, each free to track a different relationship (syntax, coreference, copying…). Cost grows with the square of sequence length — which is why every variant below exists.',
    tradeoffs:
      'MHA: every head has private K/V — best quality, biggest KV-cache. GQA: query heads share K/V in groups — near-MHA quality, much smaller cache, the modern default. MQA: one shared K/V — cheapest, some quality loss. MLA (DeepSeek): compresses K/V into a small latent, decompressed on the fly. The output gate (Qwen3-Next) multiplies the result by a learned sigmoid — it kills “attention sinks” and tolerates higher learning rates. Sparse DSA: a tiny learned indexer picks each query’s top-k most relevant past tokens and attention runs only on those — how DeepSeek V3.2 serves 128k context.',
    usedBy: 'GQA: LLaMA 3, Qwen3, Mistral. MLA: DeepSeek V2/V3, Kimi. DSA: DeepSeek V3.2, GLM-5.',
  },
  ffn: {
    title: 'Feed-Forward Network',
    body:
      'After attention gathers context, the FFN processes each token independently through a 2-layer MLP that expands to d_ff dimensions and projects back. This is where most of the model’s parameters — and most of its stored knowledge — live. SwiGLU is the modern shape: two parallel up-projections where one “gates” the other (multiplies it element-wise after a SiLU), which consistently beats plain ReLU/GELU MLPs at equal cost.',
    tradeoffs:
      'd_ff is usually ~2.7× d_model for gated activations (leave empty for auto). ReLU² (squared ReLU) is a strong cheap alternative proven at small scale by the nanoGPT speedruns. Mixture of Experts replaces this single FFN with many — see the MoE options.',
    usedBy: 'SwiGLU: LLaMA, Qwen, DeepSeek, Mistral. ReLU²: Primer, modded-nanogpt.',
  },
  moe: {
    title: 'Mixture of Experts',
    body:
      'Instead of one FFN per layer, keep N “expert” FFNs and a tiny router that sends each token to its top-k best experts (usually 2). You get the knowledge capacity of a much bigger model while each token still only pays for 2 experts of compute. The catch is load balancing: left alone, the router collapses onto a few favorite experts. The classic fix is an auxiliary loss punishing imbalance; DeepSeek-V3’s aux-loss-free trick instead nudges a per-expert bias up/down each step — no gradient interference with the main objective.',
    tradeoffs:
      'Sigmoid routing (DeepSeek-V3) scores experts independently instead of softmax-competing. A shared expert always runs alongside routed ones, absorbing common patterns so specialists can specialize. Fine-grained trend: more, thinner experts beats few big ones at equal compute.',
    usedBy: 'Mixtral (8×top-2 softmax), DeepSeek-V3 (256 experts, sigmoid + aux-free + shared), Qwen3-MoE, Kimi K2.',
  },
  residual: {
    title: 'Residual Connection',
    body:
      'Each sub-layer’s output is ADDED to its input (x + f(x)) rather than replacing it. This is the single trick that makes 100-layer networks trainable: gradients flow back through the “+” unchanged, so early layers learn even when the network is deep. It also reframes what layers do — each block only computes a small correction to the running representation (the “residual stream”), which successive layers read from and write to.',
    usedBy: 'Universal since ResNet (2015); every Transformer.',
  },
  'lm-head': {
    title: 'LM Head',
    body:
      'The final projection from d_model back to vocabulary size: one score (logit) per possible next token. A softmax turns these into probabilities, and training pushes the true next token’s probability up (cross-entropy). At generation time you sample from this distribution — temperature and top-p reshape it.',
    tradeoffs:
      'tie_embed reuses the input embedding table here (transposed) — saves vocab×d_model parameters and often trains better at small scale.',
    usedBy: 'Universal. Tied: GPT-2, Gemma, most small models.',
  },
  block: {
    title: 'Transformer Block ×N',
    body:
      'The repeating unit: normalize → mix tokens (attention) → add residual → normalize → process tokens (FFN) → add residual. Stack it N times and you have an LLM. Depth (n_layers) × width (d_model) sets capacity; typical shapes keep d_model ≈ 40-60× n_layers. Early layers tend to learn syntax and local patterns, later layers more abstract semantics.',
    tradeoffs:
      'Hybrid GDN replaces the attention in some layers with Gated DeltaNet — a linear-attention layer keeping a fixed-size memory updated with a “delta rule” (write new value, erase what was stored under that key) plus a learned decay. O(1) memory per token at inference instead of a growing KV-cache. The 3:1 pattern (3 linear : 1 full attention) is what Qwen3-Next and Kimi ship: linear layers do the bulk cheaply, periodic full-attention layers restore exact recall.',
    usedBy: 'Hybrid 3:1: Qwen3-Next, Qwen3.5, Kimi Linear/K3. Full attention: most others.',
  },
}

/** Map diagram node ids to glossary keys. */
export function glossaryKeyForNode(nodeId: string): string | null {
  if (nodeId === 'tokenizer') return 'tokenizer'
  if (nodeId === 'embed') return 'embed'
  if (nodeId === 'pos') return 'pos'
  if (nodeId === 'norm1' || nodeId === 'norm2' || nodeId === 'final-norm') return 'norm'
  if (nodeId === 'attn') return 'attn'
  if (nodeId === 'ffn') return 'ffn'
  if (nodeId.startsWith('add')) return 'residual'
  if (nodeId === 'lm-head') return 'lm-head'
  if (nodeId === 'layer-label') return 'block'
  return null
}
