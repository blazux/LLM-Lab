import { useMemo, useState } from 'react'
import ReactFlow, {
  Background,
  BackgroundVariant,
  BaseEdge,
  Controls,
  Handle,
  MarkerType,
  Panel,
  Position,
  type EdgeProps,
  type Node,
  type Edge,
  type NodeMouseHandler,
} from 'reactflow'
import 'reactflow/dist/style.css'
import type { ModelConfig, TransformerConfig } from '../types'
import { GLOSSARY, glossaryKeyForNode } from '../content/glossary'
import { estimateParams, estimateTrainVramGB, fmtParams } from '../utils/estimate'

export { estimateParams, estimateTrainVramGB }

// ─── Layout constants ─────────────────────────────────────────────────────────

const W = 296          // node width
const ROW = 27         // form row height
const HEAD = 52        // node header height
const FP = 13          // fields padding total
const GAP = 30         // standard gap between nodes
const SGAP = 62        // section gap (between input / block / output sections)
const ADD = 38         // add-node height
const PAD = 20         // block group internal padding

// Collapsed / expanded heights for the block label
const BL_COLLAPSED = 32
const BL_EXPANDED  = BL_COLLAPSED + 6 + FP + 4 * ROW   // header + divider + 4 form rows

function nh(rows: number) { return HEAD + (rows > 0 ? FP + rows * ROW : 0) }

// ─── Compact form primitives ──────────────────────────────────────────────────

const iCls = 'w-full bg-[#05080f] border border-[#1e293b] rounded px-1.5 py-0.5 text-slate-200 text-[11px] focus:outline-hidden focus:border-[#22c55e]'

function R({ l, h, children }: { l: string; h?: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, height: ROW }}>
      <span
        title={h}
        style={{
          color: '#475569', fontSize: 10, width: 82, flexShrink: 0, letterSpacing: '0.02em',
          cursor: h ? 'help' : undefined,
          textDecoration: h ? 'underline dotted #334155' : undefined,
          textUnderlineOffset: 2,
        }}
      >{l}</span>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  )
}

function II({ v, s, min }: { v: number; s: (n: number) => void; min?: number }) {
  return <input type="number" value={v} min={min} step={1} onChange={e => s(parseInt(e.target.value) || 0)} className={iCls} />
}
function FI({ v, s, step }: { v: number; s: (n: number) => void; step?: number }) {
  return <input type="number" value={v} step={step ?? 0.001} onChange={e => s(parseFloat(e.target.value) || 0)} className={iCls} />
}
function OI({ v, s, ph }: { v: number | ''; s: (n: number | '') => void; ph?: string }) {
  return <input type="number" value={v === '' ? '' : v} placeholder={ph ?? 'auto'} onChange={e => { const x = e.target.value; s(x === '' ? '' : parseFloat(x) || 0) }} className={iCls} />
}
function SI({ v, s, opts }: { v: string; s: (x: string) => void; opts: [string, string][] }) {
  return (
    <select value={v} onChange={e => s(e.target.value)} className={iCls + ' appearance-none cursor-pointer'}>
      {opts.map(([val, lbl]) => <option key={val} value={val}>{lbl}</option>)}
    </select>
  )
}
function TI({ v, s }: { v: boolean; s: (b: boolean) => void }) {
  return (
    <button type="button" onClick={() => s(!v)}
      className={`relative inline-flex h-4 w-7 shrink-0 rounded-full border-2 border-transparent transition-colors ${v ? 'bg-[#22c55e]' : 'bg-slate-700'}`}>
      <span className={`inline-block h-3 w-3 rounded-full bg-white shadow transition-transform ${v ? 'translate-x-3' : 'translate-x-0'}`} />
    </button>
  )
}

// ─── Field help texts ─────────────────────────────────────────────────────────

const HELP: Record<string, string> = {
  tokenizer: 'The tokenizer converts text into token IDs. Reusing a pretrained tokenizer (recommended) means you don’t have to train one — the vocab size below comes from it.',
  vocab_size: 'Number of distinct tokens. Set automatically from the chosen tokenizer at training start — you cannot edit it.',
  d_model: 'Width of the model: the size of every token’s hidden vector. The main knob for model capacity. Must be divisible by n_heads.',
  encoding: 'How the model knows token positions. RoPE is today’s standard; YaRN extends RoPE for longer contexts; ALiBi biases attention by distance.',
  norm_type: 'Normalization keeps activations stable. RMSNorm is the modern default (Llama, Qwen); LayerNorm is the classic choice.',
  norm_eps: 'Small constant avoiding division by zero in normalization. Leave the default unless you know why.',
  attn_type: 'MHA: every head has its own K/V (max quality, max memory). GQA: heads share K/V in groups — the modern default. MQA: all heads share one K/V. MLA: DeepSeek-style latent compression of K/V.',
  n_heads: 'Number of attention heads. Each head attends to different patterns. d_model must be divisible by n_heads.',
  n_kv_heads: 'Number of key/value heads shared across query heads (GQA). Fewer = less KV-cache memory. Must divide n_heads.',
  sliding_window: 'Limits attention to the last N tokens. Saves memory on long sequences; empty = full attention.',
  attn_bias: 'Adds bias terms to Q/K/V projections (Qwen-style). Slightly more parameters; usually off.',
  d_latent: 'MLA: dimension of the compressed KV latent. Default d_model/4.',
  d_rope_latent: 'MLA: dimension of the RoPE-carrying part of the latent. Default d_model/n_heads.',
  use_moe: 'Mixture of Experts: replaces each FFN with several “expert” FFNs and a router that picks a few per token. More capacity for the same compute per token.',
  num_experts: 'Total experts per MoE layer. More experts = more parameters (but same speed).',
  top_k: 'How many experts process each token. 2 is the common choice.',
  activation: 'FFN non-linearity. SwiGLU is the modern default (Llama, Qwen). GeGLU/ReGLU are gated variants; GELU/ReLU are classic.',
  d_ff: 'Hidden width of the feed-forward layer. Empty = auto (~2.7× d_model, the SwiGLU convention).',
  tie_embed: 'Reuse the input embedding matrix as the output projection. Saves a lot of parameters on small models; standard practice.',
  n_layers: 'Number of transformer blocks stacked. Depth × width determines total capacity.',
  max_seq_len: 'Longest sequence seen in training. Longer = more VRAM per batch.',
  dropout: 'Randomly zeroes activations during training to reduce overfitting. Modern LLM pretraining usually uses 0.',
  attn_gate: 'Sigmoid output gate on attention (Qwen3-Next). Removes attention sinks, tolerates higher learning rates. Cheap and recommended.',
  nope_every: 'Skip positional encoding every Nth layer (Llama 4 / SmolLM3 iRoPE, typically 4). NoPE layers carry global context; helps long-context. Empty = off.',
  rope_fraction: 'Fraction of head-dim pairs rotated by RoPE. 0.5 = this lab’s historical default; 0.25 = Qwen3-Next partial RoPE.',
  hybrid: 'Interleave Gated DeltaNet (linear, O(1)-memory) layers with attention layers. "3:1" = the Qwen3-Next / Kimi pattern: 3 linear then 1 attention, repeating.',
  sparse: 'DSA-style sparse attention (DeepSeek V3.2): a tiny learned indexer selects each query’s top-k tokens; attention runs only on those. GQA only.',
  sparse_top_k: 'Tokens each query may attend to. Only matters when sequences are longer than this.',
  moe_router: 'softmax = classic (Mixtral) routing with balancing losses. sigmoid = DeepSeek-V3 per-expert affinities.',
  moe_aux_free: 'DeepSeek-V3 aux-loss-free balancing: a per-expert bias steers routing toward even load, replacing the balancing losses.',
  moe_shared: 'One always-active shared expert alongside the routed ones (DeepSeek-MoE) for common knowledge.',
}

// Size presets — vocab/tokenizer untouched; params shown live since the
// embedding table (vocab × d_model) often dominates at small scale.
const SIZE_PRESETS: { id: string; label: string; patch: Partial<TransformerConfig> }[] = [
  { id: 'tiny', label: 'Tiny', patch: { d_model: 384, n_layers: 8, n_heads: 6, n_kv_heads: 2, d_ff: 1536, max_seq_len: 1024 } },
  { id: 'small', label: 'Small', patch: { d_model: 768, n_layers: 12, n_heads: 12, n_kv_heads: 4, d_ff: 2048, max_seq_len: 1024 } },
  { id: 'base', label: 'Base', patch: { d_model: 1024, n_layers: 24, n_heads: 16, n_kv_heads: 8, d_ff: 2816, max_seq_len: 2048 } },
  { id: 'large', label: 'Large', patch: { d_model: 2048, n_layers: 24, n_heads: 16, n_kv_heads: 8, d_ff: 5632, max_seq_len: 2048 } },
]

// ─── Tokenizer presets ────────────────────────────────────────────────────────

const TOKS: [string, string][] = [
  ['Qwen/Qwen2.5-0.5B', 'Qwen2.5 — 151k vocab (recommended)'],
  ['EleutherAI/gpt-neox-20b', 'GPT-NeoX — 50k vocab'],
  ['meta-llama/Llama-2-7b-hf', 'LLaMA 2 — 32k vocab'],
  ['mistralai/Mistral-7B-v0.1', 'Mistral — 32k vocab'],
  ['__custom', 'Custom path…'],
]

// ─── Field content per node ───────────────────────────────────────────────────

function fields(
  id: string,
  cfg: ModelConfig,
  set: (c: ModelConfig) => void,
): { rows: number; el: React.ReactNode } | null {
  const t = cfg as TransformerConfig

  function sT<K extends keyof TransformerConfig>(k: K, v: TransformerConfig[K]) { set({ ...t, [k]: v } as ModelConfig) }
  function sB(k: keyof ModelConfig, v: unknown) { set({ ...cfg, [k]: v } as ModelConfig) }

  const isCust = !TOKS.slice(0, -1).some(([val]) => val === cfg.tokenizer_name)

  switch (id) {
    case 'tokenizer': return { rows: 2, el: (<>
      <R l="preset" h={HELP.tokenizer}>
        <SI v={isCust ? '__custom' : cfg.tokenizer_name} s={v => { if (v !== '__custom') sB('tokenizer_name', v) }}
          opts={TOKS} />
      </R>
      <R l="path">
        <input type="text" value={cfg.tokenizer_name} onChange={e => sB('tokenizer_name', e.target.value)}
          placeholder="org/model" className={iCls + ' font-mono'} />
      </R>
    </>) }

    case 'embed': return { rows: 2, el: (<>
      <R l="vocab_size" h={HELP.vocab_size}>
        <span style={{ fontSize: 11, color: '#64748b' }}>{cfg.vocab_size.toLocaleString()} <span style={{ color: '#475569' }}>(from tokenizer)</span></span>
      </R>
      <R l="d_model" h={HELP.d_model}><II v={cfg.d_model} s={v => sB('d_model', v)} min={1} /></R>
    </>) }

    case 'pos': return { rows: 3, el: (<>
      <R l="encoding" h={HELP.encoding}>
        <SI v={t.positional_encoding} s={v => sT('positional_encoding', v as TransformerConfig['positional_encoding'])}
          opts={[['rope','RoPE'],['sinusoidal','Sinusoidal'],['alibi','ALiBi'],['yarn','YaRN']]} />
      </R>
      <R l="rope_fraction" h={HELP.rope_fraction}><FI v={t.rope_fraction} s={v => sT('rope_fraction', v)} step={0.05} /></R>
      <R l="nope_every" h={HELP.nope_every}><OI v={t.nope_every} s={v => sT('nope_every', v)} ph="off" /></R>
    </>) }

    case 'norm1':
    case 'norm2':
    case 'final-norm': return { rows: 2, el: (<>
      <R l="norm_type" h={HELP.norm_type}>
        <SI v={cfg.norm_type} s={v => sB('norm_type', v)}
          opts={[['rmsnorm','RMSNorm'],['layernorm','LayerNorm']]} />
      </R>
      <R l="norm_eps" h={HELP.norm_eps}><FI v={cfg.norm_eps} s={v => sB('norm_eps', v)} step={1e-8} /></R>
    </>) }

    case 'attn': {
      const kv = t.attention_type === 'gqa' || t.attention_type === 'mqa'
      const mla = t.attention_type === 'mla'
      const gqa = t.attention_type === 'gqa'
      const rows = 5 + (kv ? 1 : 0) + (mla ? 2 : 0) + (gqa ? 1 : 0) + (gqa && t.use_sparse_attention ? 1 : 0)
      return { rows, el: (<>
        <R l="type" h={HELP.attn_type}>
          <SI v={t.attention_type} s={v => sT('attention_type', v as TransformerConfig['attention_type'])}
            opts={[['mha','MHA — Multi-Head'],['mqa','MQA — Multi-Query'],['gqa','GQA — Grouped-Query'],['mla','MLA — Multi-Latent']]} />
        </R>
        <R l="n_heads" h={HELP.n_heads}><II v={t.n_heads} s={v => sT('n_heads', v)} min={1} /></R>
        {kv && <R l="n_kv_heads" h={HELP.n_kv_heads}><II v={t.n_kv_heads} s={v => sT('n_kv_heads', v)} min={1} /></R>}
        {mla && <>
          <R l="d_latent" h={HELP.d_latent}><OI v={t.d_latent} s={v => sT('d_latent', v)} /></R>
          <R l="d_rope_latent" h={HELP.d_rope_latent}><OI v={t.d_rope_latent} s={v => sT('d_rope_latent', v)} /></R>
        </>}
        <R l="sliding_window" h={HELP.sliding_window}><OI v={t.sliding_window} s={v => sT('sliding_window', v)} ph="none" /></R>
        <R l="attn_bias" h={HELP.attn_bias}><TI v={t.attention_bias} s={v => sT('attention_bias', v)} /></R>
        <R l="output_gate" h={HELP.attn_gate}><TI v={t.use_attention_gate} s={v => sT('use_attention_gate', v)} /></R>
        {gqa && <R l="sparse (DSA)" h={HELP.sparse}><TI v={t.use_sparse_attention} s={v => sT('use_sparse_attention', v)} /></R>}
        {gqa && t.use_sparse_attention && <R l="top_k tokens" h={HELP.sparse_top_k}><II v={t.sparse_top_k} s={v => sT('sparse_top_k', v)} min={16} /></R>}
      </>) }
    }

    case 'ffn': {
      const rows = 3 + (t.use_moe ? 6 : 0)
      return { rows, el: (<>
        <R l="use_moe" h={HELP.use_moe}><TI v={t.use_moe} s={v => sT('use_moe', v)} /></R>
        {t.use_moe && <>
          <R l="num_experts" h={HELP.num_experts}><II v={t.num_experts} s={v => sT('num_experts', v)} min={1} /></R>
          <R l="top_k" h={HELP.top_k}><II v={t.num_experts_per_token} s={v => sT('num_experts_per_token', v)} min={1} /></R>
          <R l="router" h={HELP.moe_router}>
            <SI v={t.moe_router_type} s={v => sT('moe_router_type', v as TransformerConfig['moe_router_type'])}
              opts={[['softmax','Softmax (classic)'],['sigmoid','Sigmoid (DeepSeek-V3)']]} />
          </R>
          <R l="aux-free bias" h={HELP.moe_aux_free}><TI v={t.moe_aux_free_bias} s={v => sT('moe_aux_free_bias', v)} /></R>
          <R l="shared expert" h={HELP.moe_shared}><TI v={t.moe_shared_expert} s={v => sT('moe_shared_expert', v)} /></R>
        </>}
        <R l="activation" h={HELP.activation}>
          <SI v={t.activation} s={v => sT('activation', v as TransformerConfig['activation'])}
            opts={[['swiglu','SwiGLU'],['geglu','GeGLU'],['reglu','ReGLU'],['gelu','GELU'],['relu','ReLU'],['relu2','ReLU² (Primer)'],['silu','SiLU']]} />
        </R>
        <R l="d_ff" h={HELP.d_ff}><OI v={t.d_ff} s={v => sT('d_ff', v)} ph={`auto (~${Math.round(t.d_model * 2.67)})`} /></R>
      </>) }
    }

    case 'lm-head': return { rows: 1, el: (
      <R l="tie_embed" h={HELP.tie_embed}><TI v={cfg.tie_word_embeddings} s={v => sB('tie_word_embeddings', v)} /></R>
    ) }

    case 'layer-label': return { rows: 4, el: (<>
      <R l="n_layers" h={HELP.n_layers}><II v={cfg.n_layers} s={v => sB('n_layers', v)} min={1} /></R>
      <R l="max_seq_len" h={HELP.max_seq_len}><II v={cfg.max_seq_len} s={v => sB('max_seq_len', v)} min={1} /></R>
      <R l="dropout" h={HELP.dropout}><FI v={cfg.dropout} s={v => sB('dropout', v)} step={0.01} /></R>
      <R l="hybrid GDN" h={HELP.hybrid}>
        <SI v={t.hybrid_pattern} s={v => sT('hybrid_pattern', v)}
          opts={[['','Off — all attention'],['3:1','3:1 (Qwen3-Next)'],['1:1','1:1'],['7:1','7:1']]} />
      </R>
    </>) }

    default: return null
  }
}

// ─── Node renderers ───────────────────────────────────────────────────────────

// Plain dashed frame — not a ReactFlow group, so edges route freely through it
type FrameData = { w: number; h: number; opacity: number }
function FrameNode({ data }: { data: FrameData }) {
  return (
    <div style={{
      width: data.w, height: data.h, pointerEvents: 'none',
      border: `1.5px dashed rgba(34,197,94,${data.opacity})`,
      borderRadius: 14, background: 'transparent',
    }} />
  )
}

const hDot: React.CSSProperties = { width: 10, height: 10, border: '2px solid #0a0f1c' }
const hInvis: React.CSSProperties = { width: 8, height: 8, opacity: 0, background: 'transparent', border: 'none' }
const hResVis: React.CSSProperties = { width: 9, height: 9, background: '#22c55e', border: '2px solid #060b18', borderRadius: '50%' }

type TFData = {
  label: string
  sub: string
  color: string
  fEl: React.ReactNode | null
  sel: boolean
  headerRight?: React.ReactNode
}

function TFNode({ data }: { data: TFData }) {
  return (
    <div
      style={{
        width: W,
        background: 'linear-gradient(150deg, rgba(13,20,38,0.97) 0%, rgba(6,10,22,0.99) 100%)',
        border: `1px solid ${data.sel ? data.color + '80' : '#1a2540'}`,
        borderLeft: `4px solid ${data.color}`,
        borderRadius: 10,
        boxShadow: data.sel
          ? `0 0 0 2px ${data.color}22, 0 8px 30px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.03)`
          : '0 4px 18px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.02)',
        transition: 'border-color 0.18s, box-shadow 0.18s',
        cursor: 'pointer',
        overflow: 'hidden',
      }}
    >
      <Handle type="target" position={Position.Bottom} id="bottom" style={{ ...hDot, background: data.color }} />
      <div style={{
        padding: '11px 14px 9px 12px',
        background: `linear-gradient(90deg, ${data.color}12 0%, transparent 70%)`,
        borderBottom: data.fEl ? '1px solid rgba(255,255,255,0.04)' : 'none',
        display: 'flex', alignItems: 'flex-start', gap: 8,
      }}>
        <div style={{ width: 7, height: 7, borderRadius: '50%', background: data.color, flexShrink: 0, marginTop: 3, boxShadow: `0 0 8px ${data.color}90` }} />
        <div style={{ flex: 1 }}>
          <div style={{ color: '#f1f5f9', fontSize: 12, fontWeight: 700, letterSpacing: '0.01em', lineHeight: 1.3 }}>{data.label}</div>
          {data.sub && <div style={{ color: '#475569', fontSize: 10, marginTop: 3, fontFamily: 'monospace', letterSpacing: '0.02em' }}>{data.sub}</div>}
        </div>
        {data.headerRight && <div style={{ flexShrink: 0, marginTop: 1 }}>{data.headerRight}</div>}
      </div>
      {data.fEl && (
        <div style={{ padding: '7px 12px 9px' }}
          onClick={e => e.stopPropagation()}
          onMouseDown={e => e.stopPropagation()}
          onPointerDown={e => e.stopPropagation()}>
          {data.fEl}
        </div>
      )}
      <Handle type="source" position={Position.Top} id="top" style={{ ...hDot, background: data.color }} />
      <Handle type="source" position={Position.Right} id="res-out" style={hInvis} />
    </div>
  )
}

type AddData = { sel: boolean }

function AddNode({ data }: { data: AddData }) {
  // Node is exactly ADD×ADD — wraps the circle tightly so handles land on its edge
  return (
    <div style={{ width: ADD, height: ADD, position: 'relative' }}>
      <Handle type="target" position={Position.Bottom} id="bottom" style={{ ...hDot, background: '#22c55e', bottom: -4 }} />
      <Handle type="source" position={Position.Top} id="top" style={{ ...hDot, background: '#22c55e', top: -4 }} />
      {/* res-in at top-quarter, res-out at bottom-quarter: keeps the two arcs' elbows at different Y levels */}
      <Handle type="target" position={Position.Right} id="res-in" style={{ ...hResVis, right: -4, top: '25%' }} />
      <Handle type="source" position={Position.Right} id="res-out" style={{ ...hInvis, right: -4, top: '75%' }} />
      <div style={{
        width: ADD, height: ADD, borderRadius: '50%',
        background: data.sel ? 'rgba(34,197,94,0.18)' : 'rgba(34,197,94,0.07)',
        border: `1.5px solid rgba(34,197,94,${data.sel ? '0.7' : '0.35'})`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#22c55e', fontSize: 20, fontWeight: 300, lineHeight: 1,
        boxShadow: data.sel ? '0 0 18px rgba(34,197,94,0.35)' : '0 0 10px rgba(34,197,94,0.15)',
        transition: 'all 0.15s', userSelect: 'none',
      }}>⊕</div>
    </div>
  )
}

// ─── Block label node ─────────────────────────────────────────────────────────
// Shown above the repeating block (B-T layout), displays "× N" and expands
// to show n_layers / max_seq_len / dropout when clicked.

type BlockLabelData = {
  title: string
  n_layers: number
  sel: boolean
  el: React.ReactNode | null
  blockW: number
}

function BlockLabelNode({ data }: { data: BlockLabelData }) {
  return (
    <div style={{ width: data.blockW, cursor: 'pointer' }}>
      {/* Header bar */}
      <div style={{
        height: BL_COLLAPSED,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 14px',
        background: data.sel ? 'rgba(34,197,94,0.1)' : 'rgba(34,197,94,0.05)',
        border: `1.5px solid rgba(34,197,94,${data.sel ? '0.5' : '0.22'})`,
        borderRadius: data.el ? '8px 8px 0 0' : 8,
        transition: 'background 0.15s, border-color 0.15s',
      }}>
        <span style={{ color: '#4ade80', fontSize: 9, letterSpacing: '0.12em', fontWeight: 700, textTransform: 'uppercase' as const }}>
          {data.title}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {data.sel && (
            <span style={{ color: '#475569', fontSize: 9 }}>click to collapse</span>
          )}
          <span style={{
            color: '#22c55e', fontSize: 18, fontWeight: 800, lineHeight: 1,
            textShadow: '0 0 12px rgba(34,197,94,0.6)',
          }}>
            × {data.n_layers}
          </span>
        </div>
      </div>
      {/* Expanded fields */}
      {data.el && (
        <div
          style={{
            padding: '8px 14px 10px',
            background: 'rgba(8,14,28,0.98)',
            border: '1.5px solid rgba(34,197,94,0.22)',
            borderTop: 'none',
            borderRadius: '0 0 8px 8px',
          }}
          onClick={e => e.stopPropagation()}
          onMouseDown={e => e.stopPropagation()}
          onPointerDown={e => e.stopPropagation()}
        >
          {data.el}
        </div>
      )}
    </div>
  )
}

const nodeTypes = { tfNode: TFNode, addNode: AddNode, blockLabel: BlockLabelNode, frameNode: FrameNode }

// ─── Custom residual edge ─────────────────────────────────────────────────────
// Draws a right-angle bypass to the right (→ down → ←).
// Necessary because smoothstep collapses when sourceX === targetX with same-side handles.

const RES_ARC = 60  // how far right to step

function ResEdge({ sourceX, sourceY, targetX, targetY, style, markerEnd, data }: EdgeProps) {
  const cx = W + RES_ARC + (data?.offset ?? 0)
  const midY = (sourceY + targetY) / 2
  const path = `M ${sourceX} ${sourceY} L ${cx} ${sourceY} L ${cx} ${targetY} L ${targetX} ${targetY}`
  return (
    <>
      <BaseEdge path={path} style={style} markerEnd={markerEnd} />
      <text
        x={cx + 4}
        y={midY}
        textAnchor="middle"
        dominantBaseline="middle"
        transform={`rotate(-90, ${cx + 4}, ${midY})`}
        style={{ fontSize: 9, fill: 'rgba(34,197,94,0.6)', fontWeight: 600, letterSpacing: '0.08em', userSelect: 'none', pointerEvents: 'none' }}
      >
        residual
      </text>
    </>
  )
}

const edgeTypes = { resEdge: ResEdge }

// ─── Edge builders ────────────────────────────────────────────────────────────

function mkEdge(src: string, tgt: string, color = '#334155', sh = 'top', th = 'bottom'): Edge {
  return {
    id: `${src}-${tgt}`, source: src, target: tgt, sourceHandle: sh, targetHandle: th,
    type: 'smoothstep',
    style: { stroke: color, strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12, color },
  }
}

function mkRes(src: string, tgt: string, offset = 0): Edge {
  return {
    id: `res-${src}-${tgt}`, source: src, target: tgt, sourceHandle: 'res-out', targetHandle: 'res-in',
    type: 'resEdge',
    data: { offset },
    style: { stroke: 'rgba(34,197,94,0.85)', strokeWidth: 2.5, strokeDasharray: '5 4' },
    markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12, color: 'rgba(34,197,94,0.85)' },
  }
}

// ─── Graph builders ───────────────────────────────────────────────────────────

function mkTF(id: string, label: string, sub: string, y: number, color: string, fEl: React.ReactNode | null, headerRight?: React.ReactNode): Node {
  return {
    id, position: { x: 0, y }, type: 'tfNode',
    data: { label, sub, color, fEl, sel: false, headerRight },
    draggable: false, selectable: true, connectable: false,
  }
}

function mkAdd(id: string, y: number): Node {
  // Centered horizontally so top/bottom handles align with full-width TFNodes
  return {
    id, position: { x: (W - ADD) / 2, y }, type: 'addNode',
    data: { sel: false },
    draggable: false, selectable: true, connectable: false,
  }
}

function flipNodes(nodes: Node[], heights: Map<string, number>, totalH: number): Node[] {
  return nodes.map(n => {
    const h = heights.get(n.id) ?? 0
    return { ...n, position: { x: n.position.x, y: totalH - n.position.y - h } }
  })
}

function buildTransformer(cfg: TransformerConfig, set: ((c: ModelConfig) => void) | undefined, selId: string | null): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []
  const heights = new Map<string, number>()
  const f = (id: string) => set ? fields(id, cfg, set) : null
  let y = 0

  function push(node: Node, h: number) { heights.set(node.id, h); nodes.push(node) }

  // ── Input section ──────────────────────────────────────────────────────────

  const tokF = f('tokenizer')
  const tokH = nh(selId === 'tokenizer' ? (tokF?.rows ?? 0) : 0)
  push(mkTF('tokenizer', 'Tokenizer', cfg.tokenizer_name.split('/').pop() ?? cfg.tokenizer_name, y, '#22d3ee',
    selId === 'tokenizer' ? tokF?.el ?? null : null), tokH)
  y += tokH + GAP

  const embF = f('embed')
  const embH = nh(selId === 'embed' ? (embF?.rows ?? 0) : 0)
  push(mkTF('embed', 'Token Embedding', `vocab ${cfg.vocab_size.toLocaleString()} · d_model ${cfg.d_model}`, y, '#3b82f6',
    selId === 'embed' ? embF?.el ?? null : null), embH)
  edges.push(mkEdge('tokenizer', 'embed', '#22d3ee'))
  y += embH + GAP

  const PE_SUB: Record<string, string> = {
    rope: 'RoPE — in attention',
    sinusoidal: 'Sinusoidal — added to embeddings',
    alibi: 'ALiBi — attention bias',
    yarn: 'YaRN — extended context',
  }
  const pF = f('pos')
  const pH = nh(selId === 'pos' ? (pF?.rows ?? 0) : 0)
  push(mkTF('pos', 'Positional Encoding', PE_SUB[cfg.positional_encoding] ?? cfg.positional_encoding, y, '#a855f7',
    selId === 'pos' ? pF?.el ?? null : null), pH)
  edges.push(mkEdge('embed', 'pos', '#a855f7'))
  y += pH + SGAP   // ← section gap before block

  // ── Transformer Block ──────────────────────────────────────────────────────

  const blockY = y

  const n1F = f('norm1')
  const n1H = nh(selId === 'norm1' ? (n1F?.rows ?? 0) : 0)
  push(mkTF('norm1', 'LayerNorm 1', `${cfg.norm_type} · ε=${cfg.norm_eps}`, y, '#f59e0b',
    selId === 'norm1' ? n1F?.el ?? null : null), n1H)
  edges.push(mkEdge('pos', 'norm1'))
  y += n1H + GAP

  const aF = f('attn')
  const kv = cfg.attention_type === 'gqa' || cfg.attention_type === 'mqa'
  const attnSub = `${cfg.n_heads} heads${kv ? ` · ${cfg.n_kv_heads} kv` : ''} · ${cfg.positional_encoding.toUpperCase()}`
  const aH = nh(selId === 'attn' ? (aF?.rows ?? 0) : 0)
  push(mkTF('attn', `${cfg.attention_type.toUpperCase()} Attention`, attnSub, y, '#14b8a6',
    selId === 'attn' ? aF?.el ?? null : null), aH)
  edges.push(mkEdge('norm1', 'attn', '#14b8a6'))
  y += aH + GAP

  push(mkAdd('add1', y), ADD)
  edges.push(mkEdge('attn', 'add1', '#14b8a6'))
  edges.push(mkRes('pos', 'add1'))
  y += ADD + GAP

  const n2F = f('norm2')
  const n2H = nh(selId === 'norm2' ? (n2F?.rows ?? 0) : 0)
  push(mkTF('norm2', 'LayerNorm 2', `${cfg.norm_type} · ε=${cfg.norm_eps}`, y, '#f59e0b',
    selId === 'norm2' ? n2F?.el ?? null : null), n2H)
  edges.push(mkEdge('add1', 'norm2'))
  y += n2H + GAP

  const ffF = f('ffn')
  const isMoE = cfg.use_moe
  const dff = cfg.d_ff ? cfg.d_ff : Math.round(cfg.d_model * 2.67)
  const ffnSub = isMoE
    ? `${cfg.num_experts} experts · top-${cfg.num_experts_per_token} · ${cfg.activation}`
    : `${cfg.activation} · d_ff ${dff}`
  const ffColor = isMoE ? '#ec4899' : '#f97316'
  const ffH = nh(selId === 'ffn' ? (ffF?.rows ?? 0) : 0)
  const moeBadge = set ? (
    <div
      onClick={e => { e.stopPropagation(); set({ ...cfg, use_moe: !cfg.use_moe } as ModelConfig) }}
      onMouseDown={e => e.stopPropagation()}
      onPointerDown={e => e.stopPropagation()}
      style={{
        padding: '2px 8px', borderRadius: 5, fontSize: 9, fontWeight: 700,
        background: isMoE ? 'rgba(236,72,153,0.15)' : 'rgba(71,85,105,0.2)',
        border: `1px solid ${isMoE ? 'rgba(236,72,153,0.5)' : 'rgba(71,85,105,0.4)'}`,
        color: isMoE ? '#ec4899' : '#64748b',
        cursor: 'pointer', userSelect: 'none' as const, letterSpacing: '0.04em', transition: 'all 0.15s',
      }}
    >MoE</div>
  ) : undefined
  push(mkTF('ffn', isMoE ? 'MoE FFN' : 'Feed-Forward Network', ffnSub, y, ffColor,
    selId === 'ffn' ? ffF?.el ?? null : null, moeBadge), ffH)
  edges.push(mkEdge('norm2', 'ffn', ffColor))
  y += ffH + GAP

  push(mkAdd('add2', y), ADD)
  edges.push(mkEdge('ffn', 'add2', ffColor))
  edges.push(mkRes('add1', 'add2'))
  y += ADD

  // Block geometry
  const blockH = y - blockY
  const groupH = blockH + PAD * 2
  const blockBottom = blockY + blockH

  // Ghost "stacked pages" — appear below+right of main block in B-T, suggesting repetition
  // In T-D coords: offset above the group top (= below in B-T)
  const GHOST_OFF = 8
  heights.set('ghost1', groupH)
  nodes.push({
    id: 'ghost1', type: 'frameNode',
    position: { x: -PAD + GHOST_OFF, y: blockY - PAD - GHOST_OFF },
    data: { w: W + PAD * 2, h: groupH, opacity: 0.13 },
    style: { background: 'none', border: 'none', padding: 0, boxShadow: 'none' },
    draggable: false, selectable: false, connectable: false, zIndex: -3,
  })
  heights.set('ghost2', groupH)
  nodes.push({
    id: 'ghost2', type: 'frameNode',
    position: { x: -PAD + GHOST_OFF * 2, y: blockY - PAD - GHOST_OFF * 2 },
    data: { w: W + PAD * 2, h: groupH, opacity: 0.07 },
    style: { background: 'none', border: 'none', padding: 0, boxShadow: 'none' },
    draggable: false, selectable: false, connectable: false, zIndex: -4,
  })

  // Main group box
  heights.set('layer-group', groupH)
  nodes.push({
    id: 'layer-group', type: 'frameNode',
    position: { x: -PAD, y: blockY - PAD },
    data: { w: W + PAD * 2, h: groupH, opacity: 0.28 },
    style: { background: 'none', border: 'none', padding: 0, boxShadow: 'none' },
    draggable: false, selectable: false, connectable: false, zIndex: -1,
  })

  // Block label: positioned at group bottom in T-D → appears just ABOVE the group in B-T
  const llF = f('layer-label')
  const blH = selId === 'layer-label' ? BL_EXPANDED : BL_COLLAPSED
  heights.set('layer-label', blH)
  nodes.push({
    id: 'layer-label', type: 'blockLabel',
    position: { x: -PAD, y: blockBottom + PAD },
    data: {
      title: 'Transformer Block',
      n_layers: cfg.n_layers,
      sel: selId === 'layer-label',
      el: selId === 'layer-label' ? llF?.el ?? null : null,
      blockW: W + PAD * 2,
    },
    draggable: false, selectable: true, connectable: false,
    style: { background: 'none', border: 'none', padding: 0, boxShadow: 'none' },
  })

  // Advance y past block label — ensures final-norm shifts up when label expands
  y += Math.max(SGAP, PAD + blH + GAP)

  // ── Output section ─────────────────────────────────────────────────────────

  const fnF = f('final-norm')
  const fnH = nh(selId === 'final-norm' ? (fnF?.rows ?? 0) : 0)
  push(mkTF('final-norm', 'Final LayerNorm', cfg.norm_type, y, '#f59e0b',
    selId === 'final-norm' ? fnF?.el ?? null : null), fnH)
  edges.push(mkEdge('add2', 'final-norm'))
  y += fnH + GAP

  const lmF = f('lm-head')
  const lmH = nh(selId === 'lm-head' ? (lmF?.rows ?? 0) : 0)
  push(mkTF('lm-head', 'LM Head', `Linear → vocab ${cfg.vocab_size.toLocaleString()}${cfg.tie_word_embeddings ? ' · tied' : ''}`, y, '#10b981',
    selId === 'lm-head' ? lmF?.el ?? null : null), lmH)
  edges.push(mkEdge('final-norm', 'lm-head', '#10b981'))
  const totalH = y + lmH

  const withSel = nodes.map(n => ({ ...n, data: { ...n.data, sel: n.id === selId } }))
  return { nodes: flipNodes(withSel, heights, totalH), edges }
}

// ─── Main component ───────────────────────────────────────────────────────────

interface Props {
  config: ModelConfig
  onChange?: (config: ModelConfig) => void
}

export default function ModelDiagram({ config, onChange }: Props) {
  const [selId, setSelId] = useState<string | null>(null)

  const { nodes, edges } = useMemo(() => {
    return buildTransformer(config as TransformerConfig, onChange, selId)
  }, [config, onChange, selId])

  const onNodeClick: NodeMouseHandler = (_, node) => {
    if (!onChange || node.id === 'layer-group' || node.id === 'ghost1' || node.id === 'ghost2') return
    setSelId(prev => prev === node.id ? null : node.id)
  }

  return (
    <div style={{ height: '100%', width: '100%', background: '#060b18' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.14 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag
        zoomOnScroll
        onNodeClick={onNodeClick}
        onPaneClick={() => setSelId(null)}
      >
        <Background variant={BackgroundVariant.Dots} color="#0f2040" gap={22} size={1.5} />
        <Controls showInteractive={false} style={{ bottom: 16, left: 16 }} />

        <Panel position="top-left">
          <div style={{
            background: 'rgba(6,10,22,0.95)', border: '1px solid #1a2540',
            borderRadius: 10, padding: '10px 14px', boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
            backdropFilter: 'blur(8px)', minWidth: 190,
          }}>
            <div style={{ fontSize: 10, color: '#64748b', marginBottom: 2 }}>estimated size</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: '#e2e8f0' }}>
              {fmtParams(estimateParams(config as TransformerConfig))}
              <span style={{ fontSize: 10, fontWeight: 400, color: '#64748b' }}> params</span>
            </div>
            <div
              title="Rough estimate: bf16 weights + gradients + AdamW optimizer states + checkpointed activations, batch size 1. Actual usage varies."
              style={{ fontSize: 10, color: '#64748b', cursor: 'help', textDecoration: 'underline dotted #334155', textUnderlineOffset: 2, display: 'inline-block' }}
            >
              ~{estimateTrainVramGB(config as TransformerConfig, estimateParams(config as TransformerConfig)).toFixed(1)} GB VRAM to train
            </div>
            {onChange && (
              <div style={{ display: 'flex', gap: 4, marginTop: 8 }}>
                {SIZE_PRESETS.map(p => (
                  <button
                    key={p.id}
                    onClick={() => { onChange({ ...config, ...p.patch } as ModelConfig); setSelId(null) }}
                    title={'d_model ' + p.patch.d_model + ' · ' + p.patch.n_layers + ' layers'}
                    style={{
                      padding: '4px 8px', borderRadius: 6, fontSize: 10, fontWeight: 600,
                      border: '1px solid #1e293b', background: 'transparent', color: '#64748b',
                      cursor: 'pointer', transition: 'all 0.15s',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.color = '#22c55e'; e.currentTarget.style.borderColor = 'rgba(34,197,94,0.4)' }}
                    onMouseLeave={e => { e.currentTarget.style.color = '#64748b'; e.currentTarget.style.borderColor = '#1e293b' }}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </Panel>

        {(() => {
          const key = selId ? glossaryKeyForNode(selId) : null
          const entryKey = key === 'ffn' && (config as TransformerConfig).use_moe ? 'moe' : key
          const entry = entryKey ? GLOSSARY[entryKey] : null
          if (!entry) return null
          return (
            <Panel position="bottom-center">
              <div style={{
                background: 'rgba(6,10,22,0.96)', border: '1px solid #1a2540',
                borderRadius: 12, padding: '14px 18px', maxWidth: 560, maxHeight: 260,
                overflowY: 'auto', boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
                backdropFilter: 'blur(10px)', fontSize: 12, lineHeight: 1.65,
              }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0' }}>{entry.title}</span>
                  <span style={{ fontSize: 9, color: '#475569', letterSpacing: '0.06em' }}>WHAT IT DOES</span>
                </div>
                <p style={{ color: '#94a3b8', margin: 0 }}>{entry.body}</p>
                {entry.tradeoffs && (
                  <p style={{ color: '#94a3b8', margin: '8px 0 0' }}>
                    <span style={{ color: '#eab308', fontWeight: 600 }}>Trade-offs · </span>{entry.tradeoffs}
                  </p>
                )}
                {entry.usedBy && (
                  <p style={{ color: '#64748b', margin: '8px 0 0', fontSize: 11 }}>
                    <span style={{ color: '#22c55e', fontWeight: 600 }}>In production · </span>{entry.usedBy}
                  </p>
                )}
              </div>
            </Panel>
          )
        })()}

        <Panel position="bottom-right">
          <div style={{
            background: 'rgba(6,10,22,0.9)', border: '1px solid #1a2540',
            borderRadius: 10, padding: '10px 14px', fontSize: 10, color: '#475569',
            boxShadow: '0 4px 16px rgba(0,0,0,0.4)', backdropFilter: 'blur(6px)',
            lineHeight: 1.8,
          }}>
            {onChange && <div style={{ marginBottom: 6, color: '#64748b', fontSize: 10 }}>click node to configure ↓</div>}
            {[
              ['#22d3ee', 'Tokenizer'],
              ['#3b82f6', 'Embedding'],
              ['#a855f7', 'Positional Encoding'],
              ['#f59e0b', 'Normalization'],
              ['#14b8a6', 'Attention'],
              ['#f97316', 'Feed-Forward'],
              ['#ec4899', 'MoE FFN'],
              ['#22c55e', 'Residual / Add'],
              ['#10b981', 'Output'],
            ].map(([color, label]) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, boxShadow: `0 0 5px ${color}80`, flexShrink: 0 }} />
                <span>{label}</span>
              </div>
            ))}
          </div>
        </Panel>
      </ReactFlow>
    </div>
  )
}
