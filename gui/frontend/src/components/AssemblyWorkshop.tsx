import { useCallback, useEffect, useRef, useState } from 'react'
import ReactFlow, { Background, Controls, Handle, Position, ReactFlowProvider, useReactFlow, useNodesState, useEdgesState,
  BaseEdge, MarkerType, type EdgeProps, type Connection, type NodeProps } from 'reactflow'
import 'reactflow/dist/style.css'
import type { AssemblyGraph, AssemblyNode, BrickKind, ModelConfig } from '../types'
import { sanitizeModelConfig } from '../services/api'
import ModelDiagram from './ModelDiagram'
import ArchitectureTemplates from './ArchitectureTemplates'
import AssemblySize from './AssemblySize'
import modernCatalog from '../content/modern-bricks.json'
import { conceptsForLayer } from '../content/concept-links'

const LEGACY_BRICKS: Partial<Record<BrickKind, { title: string; subtitle: string; color: string; help: string; equation: string }>> = {
  tokens: { title: 'Tokenizer / Input IDs', subtitle: 'Text → token IDs', color: '#fbbf24', help: 'The tokenizer splits text into tokens and assigns an integer ID to each token. These IDs are the input to the neural network; tokenization itself has no trainable weights here.', equation: 'text → input_ids [B, T]' },
  embedding: { title: 'Token Embedding', subtitle: 'Learned embedding lookup', color: '#818cf8', help: 'A learned table maps each token ID to a vector of size d_model. These vectors enter the first decoder block.', equation: '[B, T] → [B, T, d_model]' },
  rmsnorm: { title: 'RMSNorm', subtitle: 'Root Mean Square Normalization', color: '#2dd4bf', help: 'Normalizes the scale of each vector using its root mean square and learned weights. Placing normalization before or after a sublayer changes the computation: pre-norm and post-norm are different architectures.', equation: 'RMSNorm(x) = x / RMS(x) × weight' },
  layernorm: { title: 'LayerNorm', subtitle: 'Layer Normalization · scale + bias', color: '#2dd4bf', help: 'Centers and normalizes each token vector, then applies learned scale and bias. This is the normalization used in the original Transformer.', equation: 'LayerNorm(x) = (x − mean(x)) / √(var(x) + ε) × weight + bias' },
  position: { title: 'Positional Encoding', subtitle: 'Add sinusoidal or learned positions', color: '#fbbf24', help: 'Adds a position vector to each hidden state. Place this layer after Token Embedding for sinusoidal or learned absolute positions. RoPE instead rotates Q/K, and ALiBi biases attention scores: choose those inside Self-Attention, not as an additive layer.', equation: 'hidden_states + position_embeddings → [B, T, D]' },
  attention: { title: 'Causal Self-Attention', subtitle: 'MHA / MQA / GQA / MLA · positional method', color: '#38bdf8', help: 'Each token attends to itself and earlier tokens. MHA has independent K/V heads; MQA shares one; GQA shares groups. MLA uses a shared low-rank KV bottleneck (a simplified latent variant, without a compressed decode cache). Configure positions, a local window, Q/K normalization and output gating below.', equation: 'Attention(Q, K, V) = softmax(QKᵀ / √d_head + causal mask)V' },
  swiglu: { title: 'Feed-Forward Network', subtitle: 'FFN / MLP · configurable activation', color: '#c084fc', help: 'A position-wise feed-forward network (FFN), also called an MLP. SwiGLU, GeGLU and ReGLU use a learned gate. Standard GELU, ReLU, ReLU², SiLU and Leaky ReLU use two linear layers. The same network is applied independently to each token; it does not mix token positions.', equation: 'FFN(x) = W_down(SiLU(W_gate x) ⊙ W_up x)' },
  add: { title: 'Residual Add', subtitle: 'Element-wise addition · x + sublayer(x)', color: '#4ade80', help: 'Adds two tensors of the same shape. Connect a sublayer output to input a and the original residual stream to input b to build a residual connection. Both ports perform the same addition.', equation: 'output = a + b' },
  head: { title: 'LM Head', subtitle: 'Linear projection → vocabulary logits', color: '#fb7185', help: 'A linear projection produces one logit per vocabulary token. Training compares these logits with the next token. Softmax converts logits to probabilities when needed; it is not an extra learned layer in this graph.', equation: '[B, T, d_model] → [B, T, vocab_size]' },
  repeat: { title: 'Decoder Block × N', subtitle: 'Editable stack · independent weights', color: '#f59e0b', help: 'Open this repeated subgraph to edit its layers and residual connections. Each repetition has its own trainable weights. The example is a modern pre-norm decoder, rather than an exact reproduction of the original post-norm Transformer.', equation: 'block₁ → block₂ → … → blockₙ' },
  input_hidden: { title: 'Block Input', subtitle: 'Incoming residual stream', color: '#94a3b8', help: 'Hidden states arriving from the previous layer. Branch this stream to build a residual bypass around attention or the FFN.', equation: '[B, T, d_model]' },
  output_hidden: { title: 'Block Output', subtitle: 'Outgoing residual stream', color: '#94a3b8', help: 'Hidden states passed to the next layer. A repeated block must return the same width as its input.', equation: '[B, T, d_model]' },
}
interface ModernSetting { type: string; default: string | number | boolean; min?: number; max?: number; step?: number; options?: string[] }
const MODERN = modernCatalog as Record<string, { title: string; help: string; inputs: string[]; outputs: string[]; params: Record<string, ModernSetting> }>
const BOUNDARIES = { input_tokens: ['Token IDs Input', 'Token IDs supplied by the parent graph.'], input_kv: ['Shared KV Input', 'Key/value states from an explicitly connected earlier layer.'], input_ple: ['Per-Layer Inputs', 'Packed PLE features supplied by the parent graph.'], output_kv: ['KV Output', 'Exports key/value states for sharing with later layers.'] }
const BRICKS = { ...LEGACY_BRICKS,
  ...Object.fromEntries(Object.entries(MODERN).map(([kind,b]) => [kind, { title: b.title, subtitle: 'Executable reference brick', help: b.help, color: '#38bdf8', equation: 'Inputs → reference operation → outputs' }])),
  ...Object.fromEntries(Object.entries(BOUNDARIES).map(([kind,[title,help]]) => [kind, { title, subtitle: 'Typed block boundary', help, color: '#94a3b8', equation: 'Parent graph ↔ decoder block' }]))
} as Record<BrickKind, { title: string; subtitle: string; help: string; color: string; equation: string }>
function brickPorts(node: AssemblyNode): { inputs: string[]; outputs: string[] } {
  const kind = node.kind, p = node.params ?? {}
  if (kind === 'repeat') return { inputs: (node.body?.nodes ?? []).flatMap(n => ({ input_hidden:['in'], input_tokens:['tokens'], input_kv:['kv'], input_ple:['ple'] } as Record<string,string[]>)[n.kind] ?? []), outputs: ['out', ...((node.body?.nodes.some(n => n.kind === 'output_kv')) ? ['kv'] : [])] }
  if (MODERN[kind]) return { inputs: [...MODERN[kind].inputs, ...(kind === 'modern_attention' && p.kv_shared ? ['kv'] : []), ...(kind === 'moe' && p.family === 'gemma4' ? ['router'] : [])], outputs: MODERN[kind].outputs.filter(port => port !== 'kv' || p.family === 'gemma4') }
  if (['tokens','input_hidden','input_tokens','input_kv','input_ple'].includes(kind)) return { inputs: [], outputs: ['out'] }
  return { inputs: kind === 'add' ? ['a','b'] : ['in'], outputs: ['head','output_hidden','output_kv'].includes(kind) ? [] : ['out'] }
}
function settingVisible(node: AssemblyNode, key: string): boolean {
  if (node.kind === 'moe' && node.params?.family === 'gemma4' && key === 'shared_hidden') return false
  if (node.kind !== 'modern_attention') return true
  const p = node.params ?? {}, family = p.family ?? 'qwen35'
  if (['index_heads','index_dim','budget','compression','index_loss'].includes(key)) return family === 'qwen4'
  if (['k_eq_v','kv_shared'].includes(key)) return family === 'gemma4'
  if (key === 'temperature') return family === 'llama4' && p.position === 'none'
  if (['temperature_scale','temperature_floor'].includes(key)) return family === 'llama4' && Boolean(p.temperature) && p.position === 'none'
  if (key === 'position') return family === 'llama4' || family === 'gemma4'
  if (['rope_base','rotary_fraction'].includes(key)) return (p.position ?? 'rope') === 'rope' && !(family === 'llama4' && key === 'rotary_fraction')
  return true
}

function modernSubtitle(node: AssemblyNode): string | undefined {
  const p = node.params ?? {}
  if (!MODERN[node.kind]) return undefined
  if (node.kind === 'modern_attention') return `${p.family ?? 'qwen35'} · ${p.heads ?? 4} Q / ${p.kv_heads ?? 1} KV × ${p.head_dim ?? 16} · ${p.window ? 'local' : 'global'}${p.kv_shared ? ' · shared KV' : ''}`
  if (node.kind === 'gated_deltanet') return `${p.key_heads ?? 2} Q/K heads · ${p.value_heads ?? 4} V heads · causal conv ${p.conv_kernel ?? 4}`
  if (node.kind === 'moe') return `${p.experts ?? 8} experts · top-${p.top_k ?? 2}${p.family === 'gemma4' ? ' · parallel dense branch' : ' + shared expert'}`
  if (node.kind === 'ple_embedding') return `${p.layers ?? 5} layers × ${p.dim ?? 16} PLE values`
  if (node.kind === 'ple_apply') return `Layer ${Number(p.index ?? 0)+1} · gated PLE injection`
  if (node.kind === 'ngram') return `2–${p.ngram_size ?? 3}-gram · ${p.buckets ?? 257} base buckets · ${p.streams ?? 4} streams`
  if (['residual_expand','gr_read','gr_write','gr_collapse'].includes(node.kind)) return `${p.streams ?? 4} residual streams`
  return undefined
}

function promoteGraph(graph: AssemblyGraph): void {
  graph.version = 2
  graph.nodes.forEach(n => { if (n.body) promoteGraph(n.body) })
}
function needsModernGraph(graph: AssemblyGraph): boolean {
  return graph.version === 2 || graph.edges.some(e => e.sourceHandle && e.sourceHandle !== 'out') || graph.nodes.some(n => MODERN[n.kind] || BOUNDARIES[n.kind as keyof typeof BOUNDARIES] || n.kind === 'head' && n.params?.softcap !== undefined || n.kind === 'repeat' && (n.params?.width !== undefined || n.body && needsModernGraph(n.body)))
}

interface Inspection { valid: boolean; parameters?: number; fingerprint?: string; expanded_bricks?: number; execution_order?: string[];
  views?: Record<string, { shapes: Record<string, [string, number | null]> }>;
  shapes?: Record<string, [string, number | null]>; errors?: { message: string; node_id?: string; path?: string[]; hint?: string }[] }
interface Template { assembly: AssemblyGraph; d_model: number; n_heads: number; n_kv_heads: number; d_ff: number; n_layers: number; max_seq_len: number; dropout: number }
const clone = <T,>(value: T): T => structuredClone(value)
const button = 'rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-slate-200 hover:border-sky-400 disabled:opacity-40 disabled:cursor-not-allowed'
const input = 'w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-slate-100'

function Brick({ data, selected }: NodeProps) {
  const info = BRICKS[data.kind as BrickKind]
  const ports = data.ports as { inputs: string[]; outputs: string[] } ?? { inputs: [], outputs: [] }
  return <div style={{ width: 218, borderColor: data.error ? '#fb7185' : selected ? info.color : '#334155', borderWidth: selected ? 2 : 1 }}
    className="rounded-xl bg-slate-900 shadow-lg text-slate-100">
    {ports.inputs.map((port,i) => <Handle key={port} id={port} title={`Input: ${port}`} type="target" position={data.kind === 'add' && port === 'b' ? Position.Left : Position.Bottom} style={{ left: data.kind === 'add' && port === 'b' ? undefined : `${(i+1)/(ports.inputs.length+1)*100}%`, background: info.color, width: 12, height: 12 }} />)}
    <div className="px-4 py-3"><div className="flex justify-between gap-2"><span className="text-sm font-semibold" style={{ color: info.color }}>{data.label || info.title}</span>
      {data.kind === 'repeat' && <span className="text-xs text-amber-300">× {data.count}</span>}</div>
      <div className="mt-1 text-[10px] leading-4 text-slate-400">{data.variant || info.subtitle}</div>
      <div className="mt-2 font-mono text-[10px] text-slate-500">{data.shape || 'not connected'}</div>
      {data.kind === 'repeat' && <div className="mt-1 text-[10px] text-amber-300">Double-click to expand ↗</div>}
    </div>
    {ports.outputs.map((port,i) => <Handle key={port} id={port} title={`Output: ${port}`} type="source" position={Position.Top} style={{ left: `${(i+1)/(ports.outputs.length+1)*100}%`, background: info.color, width: 12, height: 12 }} />)}
    {(ports.inputs.length > 1 || ports.outputs.length > 1) && <div className="px-4 pb-2 text-[9px] text-slate-500">in: {ports.inputs.join(', ')} · out: {ports.outputs.join(', ')}</div>}
  </div>
}
const nodeTypes = { brick: Brick }
function ResidualEdge({ sourceX, sourceY, targetX, targetY, markerEnd, style, ...props }: EdgeProps) {
  const bypassX = Math.min(sourceX, targetX) - 140
  const path = `M ${sourceX} ${sourceY} L ${sourceX} ${sourceY - 24} L ${bypassX} ${sourceY - 24} L ${bypassX} ${targetY} L ${targetX} ${targetY}`
  return <BaseEdge id={props.id} path={path} markerEnd={markerEnd} style={style} />
}
const edgeTypes = { residual: ResidualEdge }

/** Arrange all graph levels by dependency, with input below output. */
export function arrangeBottomUp(graph: AssemblyGraph): AssemblyGraph {
  const next = clone(graph)
  const levels = new Map<string, number>()
  const pending = new Set(next.nodes.map(n => n.id))
  while (pending.size) {
    let progressed = false
    for (const id of pending) {
      const parents = next.edges.filter(e => e.target === id).map(e => e.source)
      if (parents.every(p => levels.has(p))) {
        levels.set(id, Math.max(-1, ...parents.map(p => levels.get(p)!)) + 1)
        pending.delete(id); progressed = true
      }
    }
    if (!progressed) break // Keep invalid drafts editable, including cycles and missing nodes.
  }
  const input = next.nodes.find(n => ['tokens', 'input_hidden'].includes(n.kind))
  const output = next.nodes.find(n => ['head', 'output_hidden'].includes(n.kind))
  if (input && output && (levels.get(output.id) ?? 0) <= (levels.get(input.id) ?? 0)) {
    levels.set(output.id, Math.max(1, ...levels.values()) + 1)
  }
  const max = Math.max(0, ...levels.values())
  const columns = new Map<number, number>()
  next.nodes.forEach((node, i) => {
    const level = levels.get(node.id) ?? i
    const column = columns.get(level) ?? 0
    columns.set(level, column + 1)
    node.position = { x: 160 + column * 300, y: (max - level) * 145 }
    if (node.body) node.body = arrangeBottomUp(node.body)
  })
  return next
}
function currentGraph(graph: AssemblyGraph, path: string[]): AssemblyGraph {
  for (const id of path) graph = graph.nodes.find(n => n.id === id)!.body!
  return graph
}
function attentionCount(graph: AssemblyGraph): number {
  return graph.nodes.reduce((n, b) => n + (['attention','modern_attention','gated_deltanet'].includes(b.kind) ? 1 : b.kind === 'repeat' && b.body ? Number(b.params?.count ?? 1) * attentionCount(b.body) : 0), 0)
}

function numericSettings(node: AssemblyNode, width: number): [string, string, number][] {
  const p = node.params ?? {}, size = Number(p.width ?? width)
  if (node.kind === 'attention') {
    const fields: [string, string, number][] = [['heads', 'Query heads (n_heads)', 4], ['window', 'Sliding window (0 = full causal; includes current token)', 0]]
    const mechanism = p.mechanism ?? 'gqa'
    if (mechanism === 'gqa') fields.push(['kv_heads', 'Key/value heads (n_kv_heads)', 2])
    if (mechanism === 'mla') fields.push(['latent', 'Latent KV dimension', Math.max(1, Math.floor(size / 4))])
    if ((p.position ?? 'rope') === 'rope') fields.push(['rope_fraction', 'RoPE fraction', 0.5], ['rope_base', 'RoPE base (theta)', 10000])
    return fields
  }
  if (node.kind === 'swiglu') return [['hidden', 'FFN hidden dimension (d_ff)', 4 * size]]
  if (node.kind === 'repeat') return [['count', 'Number of layers (N)', 1]]
  if (['rmsnorm', 'layernorm'].includes(node.kind)) return [['eps', 'Normalization epsilon', 1e-6]]
  if (node.kind === 'position' && (p.encoding ?? 'sinusoidal') === 'sinusoidal') return [['base', 'Sinusoidal base', 10000]]
  return []
}

function layerEquation(node: AssemblyNode): string {
  if (node.kind === 'swiglu') {
    const activation = String(node.params?.activation ?? 'swiglu')
    const gated = ['swiglu', 'geglu', 'geglu_tanh', 'reglu'].includes(activation)
    const fn = ({ swiglu: 'SiLU', geglu: 'GELU', geglu_tanh: 'GELU(tanh)', reglu: 'ReLU', relu2: 'ReLU²' } as Record<string, string>)[activation] ?? activation
    const bias = node.params?.bias ? ' · linear biases enabled' : ''
    return (gated ? `FFN(x) = down(${fn}(gate(x)) ⊙ up(x))` : `FFN(x) = down(${fn}(up(x)))`) + bias
  }
  if (node.kind === 'rmsnorm' && node.params?.unit_offset) return 'RMSNorm(x) = x / RMS(x) × (1 + weight)'
  if (node.kind === 'embedding') return node.params?.scale === 'none' ? 'input_ids → embedding lookup [B, T, d_model]' : 'input_ids → embedding lookup × √d_model'
  if (node.kind === 'attention' && node.params?.position === 'alibi') return 'softmax(QKᵀ / √d_head + ALiBi distance bias + causal mask)V'
  return BRICKS[node.kind].equation
}

function Workshop({ config, onChange }: { config: ModelConfig; onChange: (c: ModelConfig) => void }) {
  const [showTemplates, setShowTemplates] = useState(false)
  const [template, setTemplate] = useState<Template | null>(null)
  const [path, setPath] = useState<string[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [report, setReport] = useState<Inspection | null>(null)
  const [networkError, setNetworkError] = useState('')
  const [probe, setProbe] = useState('')
  const [probing, setProbing] = useState(false)
  const [tokenizerStatus, setTokenizerStatus] = useState('')
  const [readingTokenizer, setReadingTokenizer] = useState(false)
  const [history, setHistory] = useState<ModelConfig[]>([])
  const [future, setFuture] = useState<ModelConfig[]>([])
  const reference = useRef(config)
  reference.current = config
  const flow = useReactFlow()
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const graph = config.assembly ? currentGraph(config.assembly, path) : null
  const choose = graph?.nodes.find(n => n.id === selected)

  // Migrate diagrams made before the bottom-to-top convention, without changing any edges.
  useEffect(() => {
    const saved = reference.current.assembly
    if (!saved) return
    const start = saved.nodes.find(n => n.kind === 'tokens')?.position
    const end = saved.nodes.find(n => n.kind === 'head')?.position
    if (start && end && start.y < end.y) {
      onChange({ ...reference.current, assembly: arrangeBottomUp(saved) })
    }
  }, [])

  useEffect(() => { fetch('/api/assembly/template').then(r => { if (!r.ok) throw Error(); return r.json() }).then(setTemplate).catch(() => setNetworkError('The compute server is unavailable. You can edit a draft and validate it when the server returns.')) }, [])
  const replace = useCallback((next: ModelConfig) => {
    setHistory(h => [...h.slice(-39), clone(reference.current)]); setFuture([])
    reference.current = next; onChange(next); setProbe(''); setReport(null)
  }, [onChange])
  const mutate = (fn: (g: AssemblyGraph) => void) => {
    const next = clone(reference.current)
    fn(currentGraph(next.assembly!, path))
    if (needsModernGraph(next.assembly!)) promoteGraph(next.assembly!)
    next.n_layers = attentionCount(next.assembly!)
    replace(next)
  }
  useEffect(() => {
    if (!graph) return
    setEdges(graph.edges.map(e => ({ ...e, type: e.targetHandle === 'b' ? 'residual' : 'smoothstep', label: e.sourceHandle && e.sourceHandle !== 'out' ? `${e.sourceHandle} → ${e.targetHandle ?? 'in'}` : e.targetHandle && !['in','a','b'].includes(e.targetHandle) ? e.targetHandle : undefined,
      markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
      style: { stroke: '#94a3b8', strokeWidth: 2 } })))
    setNodes(graph.nodes.map((node, i) => {
      const shape = report?.views?.[path.join('/')]?.shapes[node.id] ?? (!path.length ? report?.shapes?.[node.id] : null)
      return { id: node.id, type: 'brick', position: node.position ?? { x: 160, y: (graph.nodes.length - 1 - i) * 145 },
        selected: node.id === selected, deletable: !['tokens', 'head', 'input_hidden', 'output_hidden'].includes(node.kind),
        data: { ports: brickPorts(node), label: node.label, variant: modernSubtitle(node) ?? (node.kind === 'attention' ? `${String(node.params?.mechanism ?? 'gqa').toUpperCase()} · ${String(node.params?.position ?? 'rope').toUpperCase()}${node.params?.window ? ` · window ${node.params.window}` : ''}${node.params?.gate ? ' · gated' : ''}` : node.kind === 'swiglu' ? `FFN / MLP · ${node.params?.activation ?? 'swiglu'} · hidden ${node.params?.hidden ?? 4 * Number(node.params?.width ?? config.d_model)}` : node.kind === 'position' ? `${node.params?.encoding ?? 'sinusoidal'} · additive positions` : undefined), kind: node.kind, count: node.params?.count ?? 1, shape: shape ? shape[0] === 'tokens' ? '[B, T] · token IDs' : shape[0] === 'kv' ? `KV states · ${shape[1]} values/token` : shape[0] === 'ple' ? `PLE inputs · ${shape[1]} values/token` : `[B, T, ${shape[1]}]` : null,
          error: report?.errors?.some(e => e.node_id === node.id && JSON.stringify(e.path ?? []) === JSON.stringify(path)) } }
    }))
  }, [graph, report, selected, path, setNodes])

  useEffect(() => {
    if (!config.assembly) return
    const controller = new AbortController()
    const timer = setTimeout(async () => {
      try {
        const response = await fetch('/api/assembly/inspect', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_cfg: sanitizeModelConfig(config as unknown as Record<string, unknown>) }), signal: controller.signal })
        if (!response.ok) throw Error('The server could not validate this model.')
        setReport(await response.json()); setNetworkError('')
      } catch (e) { if (!controller.signal.aborted) setNetworkError(e instanceof Error ? e.message : 'Validation unavailable') }
    }, 300)
    return () => { clearTimeout(timer); controller.abort() }
  }, [config])

  const applyTemplate = (patch: Partial<ModelConfig>) => {
    replace({ ...reference.current, ...patch, architecture: 'transformer', dropout: 0 })
    setPath([]); setSelected(null); setShowTemplates(false)
    setTimeout(() => flow.fitView({ padding: 0.3 }), 100)
  }
  const start = (guided: boolean) => {
    const assembly: AssemblyGraph = guided ? arrangeBottomUp(template!.assembly) : { version: 1, nodes: [
      { id: 'tokens', kind: 'tokens', position: { x: 150, y: 480 } }, { id: 'head', kind: 'head', position: { x: 150, y: 30 } }], edges: [] }
    replace({ ...config, ...template, assembly_template: null, assembly, d_model: 64, n_heads: 4, n_kv_heads: 2, d_ff: 128,
      n_layers: guided ? 2 : 0, max_seq_len: 256, dropout: 0, positional_encoding: 'rope', norm_type: 'rmsnorm', activation: 'swiglu',
      attention_type: 'gqa', use_moe: false, hybrid_pattern: '', use_sparse_attention: false, use_attention_gate: false, nope_every: '', tie_word_embeddings: true })
  }
  const add = (kind: BrickKind, position?: { x: number; y: number }) => {
    if (!graph || (kind === 'repeat' && !template)) return
    const id = `b_${crypto.randomUUID()}`
    const brick: AssemblyNode = { id, kind, position: position ?? flow.screenToFlowPosition({ x: window.innerWidth / 2, y: 310 }), params: MODERN[kind] ? Object.fromEntries(Object.entries(MODERN[kind].params).map(([key,spec]) => [key, key === 'width' ? config.d_model : spec.default])) : {} }
    if (kind === 'repeat') { brick.body = clone(template!.assembly.nodes.find(n => n.kind === 'repeat')!.body!); brick.params = { count: 2 } }
    mutate(g => { g.nodes.push(brick) }); setSelected(id)
  }
  const connect = (c: Connection) => {
    if (!c.source || !c.target) return
    mutate(g => { g.edges.push({ id: `e_${crypto.randomUUID()}`, source: c.source!, target: c.target!, targetHandle: c.targetHandle ?? 'in', ...(c.sourceHandle && c.sourceHandle !== 'out' ? { sourceHandle: c.sourceHandle } : {}) }) })
  }
  const remove = (ids: string[]) => mutate(g => {
    g.nodes = g.nodes.filter(n => !ids.includes(n.id) || ['tokens', 'head', 'input_hidden', 'output_hidden'].includes(n.kind))
    const kept = new Set(g.nodes.map(n => n.id)); g.edges = g.edges.filter(e => kept.has(e.source) && kept.has(e.target))
  })
  const updateSetting = (key: string, value: number | string | boolean, clear: string[] = []) => {
    if (!choose) return
    mutate(g => {
      const node = g.nodes.find(n => n.id === choose.id)!
      if (key === 'family' && MODERN[node.kind]) {
        const common = Object.fromEntries(Object.entries(node.params ?? {}).filter(([k]) => ['width','heads','kv_heads','head_dim','eps','experts','top_k','hidden','aux_weight','key_heads','value_heads','key_dim','value_dim','conv_kernel'].includes(k)))
        node.params = { ...Object.fromEntries(Object.entries(MODERN[node.kind].params).map(([k,spec]) => [k,spec.default])), ...common, family:value }
        if (node.kind === 'modern_attention' && value === 'llama4') node.params.rotary_fraction = 1
      } else node.params = { ...node.params, [key]: value }
      clear.forEach(k => delete node.params![k])
    })
  }
  const selectSetting = (key: string, label: string, fallback: string, options: string[][], clear: string[] = []) => <label className="mt-3 block text-xs text-slate-400">{label}<select className={input + ' mt-1'} value={String(choose?.params?.[key] ?? fallback)} onChange={e => updateSetting(key, e.target.value, clear)}>{options.map(([value, text]) => <option key={value} value={value}>{text}</option>)}</select></label>
  const checkSetting = (key: string, label: string, fallback: boolean) => <label className="mt-3 flex items-center gap-2 text-xs text-slate-400"><input type="checkbox" checked={Boolean(choose?.params?.[key] ?? fallback)} onChange={e => updateSetting(key, e.target.checked)} />{label}</label>
  const undo = () => { if (!history.length) return; const next = history[history.length - 1]; setFuture(f => [clone(config), ...f]); setHistory(h => h.slice(0, -1)); reference.current = next; onChange(next); setPath([]); setSelected(null); setReport(null); setProbe('') }
  const redo = () => { if (!future.length) return; const next = future[0]; setHistory(h => [...h, clone(config)]); setFuture(f => f.slice(1)); reference.current = next; onChange(next); setPath([]); setSelected(null); setReport(null); setProbe('') }
  const test = async () => {
    const tested = JSON.stringify(config)
    setProbing(true); setProbe('')
    try {
      const response = await fetch('/api/assembly/probe', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_cfg: sanitizeModelConfig(config as unknown as Record<string, unknown>) }) })
      const data = await response.json()
      if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : data.detail?.message ?? 'The computation failed.')
      if (JSON.stringify(reference.current) !== tested) setProbe('The model changed during the test. Run it again to check this version.')
      else setProbe(`${data.message} Test ran on ${data.device}.`)
    } catch (e) { setProbe(e instanceof Error ? e.message : 'Test failed') }
    finally { setProbing(false) }
  }
  const readTokenizer = async () => {
    const name = config.tokenizer_name
    setReadingTokenizer(true); setTokenizerStatus('')
    try {
      const response = await fetch('/api/assembly/tokenizer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) })
      const data = await response.json()
      if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Could not read tokenizer')
      if (reference.current.tokenizer_name !== name) return
      replace({ ...reference.current, vocab_size: data.vocab_size })
      setTokenizerStatus(`${data.vocab_size.toLocaleString()} tokens read from tokenizer.`)
    } catch (error) { setTokenizerStatus(error instanceof Error ? error.message : 'Could not read tokenizer') }
    finally { setReadingTokenizer(false) }
  }

  if (!graph) return <div className="h-full overflow-auto bg-slate-950 p-8 text-slate-100">
    <div className="mx-auto max-w-4xl"><span className="text-xs uppercase tracking-[.25em] text-sky-400">Model workshop · learn by building</span>
      <h1 className="mt-4 text-3xl font-semibold">Build your own model.</h1>
      <p className="mt-3 max-w-2xl leading-7 text-slate-400">Connect layers, expand blocks and try your ideas. The engine executes the graph you build. Undo a change, make mistakes and try again.</p>
      <div className="mt-8 grid gap-4 md:grid-cols-2"><button className="rounded-2xl border border-sky-800 bg-sky-950/40 p-6 text-left hover:border-sky-400" onClick={() => start(false)}>
        <span className="text-xl font-semibold">Start from a blank canvas ↗</span><p className="mt-3 text-sm leading-6 text-slate-400">Input IDs at the bottom, LM Head at the top. Add a Token Embedding and connect your layers between them.</p></button>
        <button className="rounded-2xl border border-slate-700 bg-slate-900 p-6 text-left hover:border-amber-400 disabled:opacity-40" disabled={!template} onClick={() => start(true)}>
        <span className="text-xl font-semibold">Explore a small Transformer ↗</span><p className="mt-3 text-sm leading-6 text-slate-400">A pre-norm decoder with d_model = 64 and 2 blocks. Explore the residual paths, remove a layer and observe what changes.</p></button></div>
      <div className="mt-6"><ArchitectureTemplates onApply={applyTemplate} /></div>
      {networkError && <p role="status" className="mt-4 text-sm text-amber-300">{networkError}</p>}
      {!!history.length && <button className={button + ' mt-4'} onClick={undo}>Undo · restore the previous model</button>}
      <details className="mt-8 rounded-xl border border-slate-800 p-4"><summary className="cursor-pointer text-sm text-slate-400">Existing model · read-only until replaced</summary><div className="mt-3 h-[480px]"><ModelDiagram config={config} /></div></details>
    </div></div>

  return <div className="flex h-full min-h-0 flex-col bg-[#080d18] text-slate-100">
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-5 py-3">
      <div><div className="text-xs uppercase tracking-widest text-sky-400">Model Assembly Workshop</div><div className="mt-1 text-xs text-slate-400">Data flows upward. Connections define the model; positions organize the diagram.</div></div>
      <div className="flex flex-wrap gap-2"><button className={button} onClick={() => setShowTemplates(v => !v)}>Architecture templates</button><button className={button} onClick={() => { mutate(g => Object.assign(g, arrangeBottomUp(g))); setTimeout(() => flow.fitView({ padding: 0.3 }), 100) }}>↑ Arrange layers</button><button className={button} disabled={!history.length} onClick={undo}>↶ Undo</button><button className={button} disabled={!future.length} onClick={redo}>↷ Redo</button>
        <button className={button + ' border-sky-700 text-sky-200'} disabled={!report?.valid || probing} onClick={test}>{probing ? 'Running…' : '▷ Test forward & backward'}</button></div>
    </header>
    {showTemplates && <div className="max-h-80 overflow-auto p-4"><ArchitectureTemplates onApply={applyTemplate} /></div>}
    <div className="flex min-h-0 flex-1">
      <aside className="w-52 flex-none space-y-2 overflow-auto border-r border-slate-800 p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">Layers & blocks</h2>
        <p className="mb-4 text-[11px] leading-5 text-slate-500">Drag a layer onto the canvas, or click to add it.</p>
        {(['embedding', 'position', 'rmsnorm', 'layernorm', 'attention', 'swiglu', 'add', 'repeat', ...Object.keys(MODERN), ...(path.length ? Object.keys(BOUNDARIES) : [])] as BrickKind[]).filter(kind => !(path.length && kind === 'embedding')).map(kind => <button key={kind} draggable onDragStart={e => e.dataTransfer.setData('application/llmlab-brick', kind)}
          className="block w-full rounded-xl border border-slate-800 bg-slate-900 px-3 py-3 text-left hover:border-slate-500 disabled:opacity-40" disabled={kind === 'repeat' && !template} onClick={() => add(kind)}>
          <span style={{ color: BRICKS[kind].color }} className="text-sm font-semibold">＋ {BRICKS[kind].title}</span><span className="mt-1 block text-[10px] leading-4 text-slate-500">{BRICKS[kind].subtitle}</span></button>)}
        <div className="pt-4 text-[11px] leading-5 text-slate-500">Connect the top output of one layer to the bottom input of the next. Residual Add has two inputs; b is the bypass.</div>
      </aside>
      <main className="relative min-w-0 flex-1" onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy' }} onDrop={e => { e.preventDefault(); const kind = e.dataTransfer.getData('application/llmlab-brick') as BrickKind; if (BRICKS[kind]) add(kind, flow.screenToFlowPosition({ x: e.clientX, y: e.clientY })) }}>
        <div className="absolute left-4 top-3 z-10 flex gap-2"><button className={button} onClick={() => { setPath([]); setSelected(null) }}>Model</button>
          {path.map((id, i) => <button key={i} className={button + ' text-amber-300'} onClick={() => { setPath(path.slice(0, i + 1)); setSelected(null) }}>{currentGraph(config.assembly!, path.slice(0,i)).nodes.find(n => n.id === id)?.label ?? `Block ${i+1}`}</button>)}</div>
        <ReactFlow nodes={nodes} edges={edges}
          nodeTypes={nodeTypes} edgeTypes={edgeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={connect}
          onNodesDelete={deleted => remove(deleted.map(n => n.id))} onEdgesDelete={deleted => mutate(g => { const ids = new Set(deleted.map(e => e.id)); g.edges = g.edges.filter(e => !ids.has(e.id)) })}
          onNodeClick={(_, n) => setSelected(n.id)} onNodeDoubleClick={(_, n) => { if (graph.nodes.find(b => b.id === n.id)?.kind === 'repeat') { setPath([...path, n.id]); setSelected(null); setTimeout(() => flow.fitView({ padding: 0.25 }), 100) } }}
          onNodeDragStop={(_, n) => mutate(g => { g.nodes.find(b => b.id === n.id)!.position = n.position })}
          fitView fitViewOptions={{ padding: 0.3 }} minZoom={0.15} maxZoom={1.8} deleteKeyCode={['Backspace', 'Delete']}>
          <Background color="#1e293b" gap={24} /><Controls />
        </ReactFlow>
      </main>
      <aside className="w-72 flex-none space-y-5 overflow-auto border-l border-slate-800 bg-slate-950/60 p-4">
        <section><h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Your model</h2>
          {config.assembly_template && <p className="mt-2 text-xs text-slate-400">Starting point: {({ llama3: 'Llama 3', gemma1: 'Gemma 1', qwen25: 'Qwen 2.5', qwen35_dense:'Qwen 3.5 dense', qwen35_moe:'Qwen 3.5 MoE', llama4_scout:'Llama 4 Scout', llama4_maverick:'Llama 4 Maverick', gemma4_e2b:'Gemma 4 E2B', gemma4_e4b:'Gemma 4 E4B', gemma4_31b:'Gemma 4 31B', gemma4_moe:'Gemma 4 26B-A4B', qwen4_preview:'Qwen4 preview / Qwen3.8-Flash-Next' } as Record<string, string>)[config.assembly_template] ?? config.assembly_template} · editable</p>}
          {config.assembly?.version === 2 && <p className="mt-2 text-[11px] leading-5 text-amber-300">Experimental graph · training validation pending.</p>}
          <AssemblySize key={`${config.d_model}-${config.n_layers}-${config.max_seq_len}-${config.assembly_template}`} config={config} onApply={next => { replace(next); setPath([]); setSelected(null) }} />
          <label className="mt-3 block text-xs text-slate-400">Model dimension (d_model)<input className={input + ' mt-1'} aria-label="Model dimension" type="number" min={1} value={config.d_model} onChange={e => replace({ ...config, d_model: Number(e.target.value) })} /></label>
          <label className="mt-3 block text-xs text-slate-400">Context length (tokens)<input className={input + ' mt-1'} type="number" min={2} value={config.max_seq_len} onChange={e => replace({ ...config, max_seq_len: Number(e.target.value) })} /></label>
          <label className="mt-3 flex items-start gap-2 text-xs text-slate-400"><input type="checkbox" checked={config.tie_word_embeddings} onChange={e => replace({ ...config, tie_word_embeddings: e.target.checked })} />Tie Token Embedding and LM Head weights</label>
          {report?.valid && <div className="mt-3 rounded-lg border border-emerald-900 bg-emerald-950/30 p-3 text-xs leading-5 text-emerald-200"><strong>{report.parameters?.toLocaleString()} parameters</strong><br />Graph validated · {report.expanded_bricks} expanded operations<br /><span className="font-mono text-[10px] text-emerald-500">graph {report.fingerprint}</span><p className="mt-1 text-[10px] text-slate-400">Current vocab_size: {config.vocab_size.toLocaleString()}. Read the tokenizer in Input IDs to update this count.</p></div>}
        </section>
        <section aria-live="polite"><h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Understand & experiment</h2>
          {networkError ? <p className="mt-3 text-xs leading-5 text-amber-300">{networkError}</p> : report?.valid === false ? report.errors?.map((error, i) => <div key={i} className="mt-3 rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-xs leading-5">
            <p className="text-rose-200">{error.message}</p><p className="mt-2 text-slate-400">{error.hint}</p>{error.node_id && <button className={button + ' mt-2'} onClick={() => { setPath(error.path ?? []); setSelected(error.node_id!); setTimeout(() => flow.fitView({ padding: 0.25 }), 100) }}>Show affected layer</button>}</div>) : <p className="mt-3 text-xs leading-5 text-slate-400">{report ? 'The graph is valid. Test it, then change one thing to observe its effect.' : 'Validating graph…'}</p>}
          {probe && <p role="status" className="mt-3 rounded-lg border border-slate-700 p-3 text-xs leading-5 text-sky-200">{probe}</p>}
        </section>
        {choose ? <section className="border-t border-slate-800 pt-4"><h2 style={{ color: BRICKS[choose.kind].color }} className="font-semibold">{BRICKS[choose.kind].title}</h2><p className="mt-2 text-xs leading-6 text-slate-400">{BRICKS[choose.kind].help}</p><p className="mt-3 rounded-lg bg-slate-900 p-2 font-mono text-[10px] leading-5 text-slate-400">{layerEquation(choose)}</p>
          {conceptsForLayer(choose).length > 0 && <nav aria-label="Understand this layer" className="mt-3 flex flex-col gap-2 text-xs">
            {conceptsForLayer(choose).map(concept => <a key={concept.id} href={concept.url} target="_blank" rel="noopener noreferrer" className="text-sky-300 underline decoration-sky-800 underline-offset-4 hover:text-sky-100">{concept.title} ↗</a>)}
          </nav>}
          {choose.kind === 'tokens' && <div className="mt-3 space-y-2"><label className="block text-xs text-slate-400">Local tokenizer or Hugging Face ID<input className={input + ' mt-1'} value={config.tokenizer_name} onChange={e => { replace({ ...config, tokenizer_name: e.target.value }); setTokenizerStatus('') }} /></label><button className={button} disabled={readingTokenizer} onClick={readTokenizer}>{readingTokenizer ? 'Reading…' : 'Read tokenizer'}</button><p role="status" className="text-xs leading-5 text-slate-400">{tokenizerStatus || 'This uses an existing tokenizer. Its vocabulary determines the size of the Token Embedding and LM Head.'}</p></div>}
          {['embedding', 'position', 'rmsnorm', 'layernorm', 'attention', 'swiglu', 'head'].includes(choose.kind) && <label className="mt-3 block text-xs text-slate-400">Dimension (empty = d_model)<input className={input + ' mt-1'} type="number" min={1} value={choose.params?.width === undefined ? '' : Number(choose.params.width)} placeholder={String(config.d_model)} onChange={e => mutate(g => { const node = g.nodes.find(n => n.id === choose.id)!; node.params ??= {}; if (e.target.value === '') delete node.params.width; else node.params.width = Number(e.target.value) })} /></label>}
          {choose.kind === 'embedding' && selectSetting('scale', 'Embedding scaling', 'sqrt_dim', [['sqrt_dim', 'Multiply by √d_model'], ['none', 'No scaling']])}
          {choose.kind === 'rmsnorm' && checkSetting('unit_offset', 'Unit-offset weights (Gemma: 1 + weight)', false)}
          {choose.kind === 'swiglu' && <>
            {selectSetting('activation', 'FFN activation', 'swiglu', [['swiglu', 'SwiGLU'], ['geglu', 'GeGLU'], ['geglu_tanh', 'GeGLU (tanh GELU)'], ['reglu', 'ReGLU'], ['gelu', 'GELU'], ['relu', 'ReLU'], ['relu2', 'ReLU²'], ['silu', 'SiLU'], ['leaky_relu', 'Leaky ReLU']])}
            {checkSetting('bias', 'Linear biases', false)}
            <p className="mt-2 text-[11px] leading-5 text-slate-500">Gated FFNs have three weight matrices; standard FFNs have two. This changes both computation and parameter count.</p>
          </>}
          {choose.kind === 'position' && <>
            {selectSetting('encoding', 'Absolute position encoding', 'sinusoidal', [['sinusoidal', 'Sinusoidal (fixed)'], ['learned', 'Learned position embeddings']], ['base'])}
            <p className="mt-2 text-[11px] leading-5 text-slate-500">For an absolute-position baseline, set each attention layer’s positional method to None. Combining position methods is allowed as an explicit experiment.</p>
          </>}
          {choose.kind === 'attention' && <>
            {selectSetting('backend', 'Attention execution backend', 'auto', [['auto', 'Auto — PyTorch SDPA dispatch'], ['math', 'Math reference']])}
            {selectSetting('mechanism', 'Attention mechanism', 'gqa', [['mha', 'MHA — Multi-Head Attention'], ['mqa', 'MQA — Multi-Query Attention'], ['gqa', 'GQA — Grouped-Query Attention'], ['mla', 'MLA — latent KV compression']], ['kv_heads', 'latent'])}
            {selectSetting('position', 'Attention positional method', 'rope', [['rope', 'RoPE — rotary Q/K positions'], ['alibi', 'ALiBi — attention distance bias'], ['none', 'None — no attention position method']], ['rope_fraction', 'rope_base', 'rope_style'])}
            {(choose.params?.position ?? 'rope') === 'rope' && selectSetting('rope_style', 'RoPE frequency convention', 'legacy', [['legacy', 'Legacy LLM-Lab'], ['standard', 'Standard inverse frequencies']])}
            {checkSetting('qk_norm', 'Q/K RMSNorm', true)}
            {checkSetting('qkv_bias', 'Q/K/V projection biases', false)}
            {checkSetting('gate', 'Sigmoid output gate', false)}
            <p className="mt-2 text-[11px] leading-5 text-slate-500">RoPE rotates Q/K inside attention; ALiBi biases scores. Neither adds vectors to Token Embedding. YaRN context extension is not a base-model layer.</p>
          </>}
          {MODERN[choose.kind] && <div className="mt-3 space-y-3">{Object.entries(MODERN[choose.kind].params).filter(([key]) => settingVisible(choose,key)).map(([key,spec]) => spec.type === 'select' ? selectSetting(key, key.replace(/_/g,' '), String(spec.default), (spec.options ?? []).map(value => [value,value])) : spec.type === 'boolean' ? checkSetting(key,key.replace(/_/g,' '),Boolean(spec.default)) : <label key={key} className="block text-xs text-slate-400">{key.replace(/_/g,' ')}<input className={input + ' mt-1'} type="number" min={spec.min} max={spec.max} step={spec.step} value={Number(choose.params?.[key] ?? (key === 'width' ? config.d_model : spec.default))} onChange={e => updateSetting(key,Number(e.target.value))} /></label>)}</div>}
          {['input_kv','input_ple'].includes(choose.kind) && <label className="mt-3 block text-xs text-slate-400">Boundary dimension<input className={input + ' mt-1'} type="number" min={1} value={Number(choose.params?.width ?? config.d_model)} onChange={e => updateSetting('width',Number(e.target.value))} /></label>}
          {choose.kind === 'head' && config.assembly?.version === 2 && <label className="mt-3 block text-xs text-slate-400">Logit softcap (0 = disabled)<input className={input + ' mt-1'} type="number" min={0} value={Number(choose.params?.softcap ?? 0)} onChange={e => updateSetting('softcap',Number(e.target.value))} /></label>}
          {numericSettings(choose, config.d_model).map(([key, label, fallback]) => <label key={key} className="mt-3 block text-xs text-slate-400">{label}<input className={input + ' mt-1'} type="number" step={key === 'rope_fraction' ? 0.05 : key === 'eps' ? 'any' : 1} value={Number(choose.params?.[key] ?? fallback)} onChange={e => updateSetting(key, Number(e.target.value))} /></label>)}
          {choose.kind === 'repeat' && <button className={button + ' mt-3 w-full'} onClick={() => { setPath([...path, choose.id]); setSelected(null); setTimeout(() => flow.fitView({ padding: 0.25 }), 100) }}>Expand & edit block ↗</button>}
          {!['tokens', 'head', 'input_hidden', 'output_hidden'].includes(choose.kind) && <div className="mt-3 flex gap-2"><button className={button} onClick={() => { const copied = clone(choose); copied.id = `b_${crypto.randomUUID()}`; copied.position = { x: (choose.position?.x ?? 160) + 260, y: choose.position?.y ?? 100 }; mutate(g => g.nodes.push(copied)); setSelected(copied.id) }}>Duplicate</button><button className={button + ' text-rose-300'} onClick={() => { remove([choose.id]); setSelected(null) }}>Remove</button></div>}
        </section> : <p className="text-xs leading-6 text-slate-500">Select a layer to learn its role and adjust its parameters. B = batch, T = tokens, D = model dimension.</p>}
      </aside>
    </div>
  </div>
}

export default function AssemblyWorkshop(props: { config: ModelConfig; onChange: (c: ModelConfig) => void }) {
  return <ReactFlowProvider><Workshop {...props} /></ReactFlowProvider>
}
