import type { TransformerConfig, AssemblyGraph } from '../types'
import { modernParameters } from './modernEstimate'

function assemblyParams(graph: AssemblyGraph, width: number, vocab: number, tied: boolean, context: number): number {
  return graph.nodes.reduce((total, node) => {
    const p = node.params ?? {}, w = Number(p.width ?? width)
    const modern = modernParameters(node,width,vocab)
    if (modern !== null) return total + modern
    if (node.kind === 'embedding') return total + vocab * w
    if (node.kind === 'head') return total + (tied ? 0 : vocab * w)
    if (node.kind === 'rmsnorm') return total + w
    if (node.kind === 'layernorm') return total + 2 * w
    if (node.kind === 'position') return total + (p.encoding === 'learned' ? context * w : 0)
    if (node.kind === 'swiglu') {
      const hidden = Number(p.hidden ?? 4 * w), gated = ['swiglu', 'geglu', 'geglu_tanh', 'reglu'].includes(String(p.activation ?? 'swiglu'))
      return total + (gated ? 3 : 2) * w * hidden + (p.bias ? (gated ? 2 : 1) * hidden + w : 0)
    }
    if (node.kind === 'attention') {
      const heads = Number(p.heads ?? 4), dk = w / Math.max(1, heads), mechanism = p.mechanism ?? 'gqa'
      const kv = mechanism === 'mha' || mechanism === 'mla' ? heads : mechanism === 'mqa' ? 1 : Number(p.kv_heads ?? 2)
      const latent = Number(p.latent ?? Math.max(1, Math.floor(w / 4)))
      const projections = mechanism === 'mla' ? 2 * w * w + w * latent + 2 * latent * w : 2 * w * w + 2 * w * kv * dk
      return total + projections + (p.qk_norm === false ? 0 : 2 * dk) + (p.gate ? w * w : 0) + (p.qkv_bias ? w + 2 * kv * dk : 0)
    }
    if (node.kind === 'repeat' && node.body) return total + Number(p.count ?? 1) * assemblyParams(node.body, w, vocab, false, context)
    return total
  }, 0)
}

// ─── Parameter estimation ─────────────────────────────────────────────────────

export function estimateParams(cfg: TransformerConfig): number {
  if (cfg.assembly) return assemblyParams(cfg.assembly, cfg.d_model, cfg.vocab_size, cfg.tie_word_embeddings, cfg.max_seq_len)
  const d = cfg.d_model
  const dk = Math.floor(d / Math.max(cfg.n_heads, 1))
  const dff = cfg.d_ff === '' ? Math.round(d * 8 / 3) : cfg.d_ff
  const embed = cfg.vocab_size * d

  let attn: number
  if (cfg.attention_type === 'mha') attn = 4 * d * d
  else if (cfg.attention_type === 'mqa') attn = d * d + 2 * d * dk + d * d
  else if (cfg.attention_type === 'mla') {
    const dl = cfg.d_latent === '' ? Math.max(Math.floor(d / 4), dk) : cfg.d_latent
    attn = d * d + d * dl + 2 * dl * d + d * d
  } else attn = d * d + 2 * cfg.n_kv_heads * dk * d + d * d

  if (cfg.use_attention_gate) attn += d * d

  if (cfg.use_sparse_attention && cfg.attention_type === 'gqa') {
    const ih = 4, id_ = 32
    attn += d * ih * id_ + d * id_ + d * ih
  }

  const gated = ['swiglu', 'geglu', 'geglu_tanh', 'reglu'].includes(cfg.activation)
  const ffnOne = (gated ? 3 : 2) * d * dff
  let ffn = cfg.use_moe ? ffnOne * cfg.num_experts + d * cfg.num_experts : ffnOne
  if (cfg.use_moe && cfg.moe_shared_expert) ffn += ffnOne

  // Hybrid: Gated DeltaNet layers replace attention params on linear layers
  const gdn = 5 * d * d + 2 * d * cfg.n_heads
  let nLinear = 0
  if (cfg.hybrid_pattern) {
    const [lin, att] = cfg.hybrid_pattern.split(':').map(Number)
    if (lin > 0 && att > 0) {
      const cycle = lin + att
      for (let i = 0; i < cfg.n_layers; i++) if (i % cycle < lin) nLinear++
    }
  }

  const attnLayer = attn + ffn + 4 * d
  const gdnLayer = gdn + ffn + 4 * d
  const layers = nLinear * gdnLayer + (cfg.n_layers - nLinear) * attnLayer
  return embed + layers + d + (cfg.tie_word_embeddings ? 0 : embed)
}

export function fmtParams(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (n >= 1e6) return (n / 1e6).toFixed(0) + 'M'
  return (n / 1e3).toFixed(0) + 'k'
}

/** Rough pretraining VRAM: fp32 weights+grads (8 B/param) + fp32 AdamW moments (8 B/param)
 *  + checkpointed activations. Indicative only. */
export function estimateTrainVramGB(cfg: TransformerConfig, params: number): number {
  const states = params * 16
  const activations = cfg.max_seq_len * cfg.d_model * cfg.n_layers * 2 * 6
  return (states + activations) / 1024 ** 3
}

// ─── Detailed training-memory breakdown ──────────────────────────────────────

export interface TrainMemoryEstimate {
  params: number
  trainableParams: number
  weightsGB: number
  gradsGB: number
  optimizerGB: number
  activationsGB: number
  logitsGB: number
  overheadGB: number
  totalGB: number
}

export interface TrainMemoryOpts {
  batchSize: number
  optimizer: string
  /** pretrain loop always checkpoints activations; the SFT loop does not */
  gradientCheckpointing: boolean
  useLora?: boolean
  loraR?: number
  masterWeights?: boolean
}

/** Indicative memory estimate. Kernels, graph topology, fragmentation and
 *  precision choices can change the actual peak substantially. */
export function estimateTrainMemory(cfg: TransformerConfig, opts: TrainMemoryOpts): TrainMemoryEstimate {
  const P = estimateParams(cfg)
  const B = Math.max(1, opts.batchSize || 1)
  const T = cfg.max_seq_len
  const d = cfg.d_model
  const L = cfg.n_layers
  const V = cfg.vocab_size

  // LoRA: only adapter matrices get grads + optimizer states
  // (≈ 4 adapted projections per layer, each two r×d factors)
  const trainable = opts.useLora ? L * 4 * 2 * (opts.loraR ?? 8) * d : P

  const weights = (opts.masterWeights ? 4 : 2) * P
  const grads = (opts.masterWeights ? 4 : 2) * trainable
  const bytesPerState = opts.optimizer === 'lion' || opts.optimizer === 'muon' ? 4 : 8
  const optimizer = bytesPerState * trainable // fp32 moments

  // bf16 activations: with checkpointing only block inputs survive
  // (+ one block being recomputed); without it, every intermediate does
  const activations = opts.gradientCheckpointing
    ? B * T * d * 2 * (L + 12)
    : B * T * d * 2 * L * 16

  // logits (bf16) + fp32 log-softmax + fp32 grad over the full vocabulary —
  // often the biggest single tensor with a 150k-vocab tokenizer
  const logits = B * T * V * 10

  const overhead = 1.2 * 1024 ** 3 // CUDA context + allocator slack

  const GB = (x: number) => x / 1024 ** 3
  return {
    params: P,
    trainableParams: trainable,
    weightsGB: GB(weights),
    gradsGB: GB(grads),
    optimizerGB: GB(optimizer),
    activationsGB: GB(activations),
    logitsGB: GB(logits),
    overheadGB: GB(overhead),
    totalGB: GB(weights + grads + optimizer + activations + logits + overhead),
  }
}
