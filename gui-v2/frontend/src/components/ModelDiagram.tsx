import { useMemo } from 'react'
import ReactFlow, {
  Background,
  Controls,
  BackgroundVariant,
  type Node,
  type Edge,
} from 'reactflow'
import 'reactflow/dist/style.css'
import type { ModelConfig, TransformerConfig, Mamba2Config } from '../types'

// ─── Shared node builder ──────────────────────────────────────────────────────

function makeNode(
  id: string,
  label: string,
  sublabel: string,
  x: number,
  y: number,
  borderColor: string,
  bgColor = '#1e293b'
): Node {
  return {
    id,
    position: { x, y },
    data: { label, sublabel, borderColor, bgColor },
    type: 'diagramNode',
    draggable: false,
    selectable: false,
    connectable: false,
  }
}

function makeEdge(source: string, target: string, color = '#334155'): Edge {
  return {
    id: `${source}-${target}`,
    source,
    target,
    type: 'straight',
    style: { stroke: color, strokeWidth: 1.5 },
    animated: false,
  }
}

// ─── Transformer graph ────────────────────────────────────────────────────────

function buildTransformerGraph(cfg: TransformerConfig): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []

  const cx = 100
  let y = 0
  const gap = 90

  // Token Embedding
  nodes.push(makeNode('embed', 'Token Embedding', `vocab=${cfg.vocab_size.toLocaleString()}`, cx, y, '#3b82f6', '#1e293b'))
  y += gap

  // Positional Encoding (only if explicit)
  const needsExplicitPE = cfg.positional_encoding === 'sinusoidal'
  if (needsExplicitPE) {
    nodes.push(makeNode('pos', 'Positional Encoding', cfg.positional_encoding, cx, y, '#a855f7', '#1e293b'))
    edges.push(makeEdge('embed', 'pos', '#a855f7'))
    y += gap
  }

  // Layer block label
  const blockTopY = y
  const blockPad = 12
  const innerGap = 72

  let prevInBlock = needsExplicitPE ? 'pos' : 'embed'

  // Norm1
  nodes.push(makeNode('norm1', 'LayerNorm', cfg.norm_type, cx, y, '#f59e0b'))
  edges.push(makeEdge(prevInBlock, 'norm1'))
  y += innerGap
  prevInBlock = 'norm1'

  // Attention
  const attnLabel = cfg.attention_type.toUpperCase()
  const attnSub = `heads=${cfg.n_heads}${cfg.attention_type === 'gqa' || cfg.attention_type === 'mqa' ? ` kv=${cfg.n_kv_heads}` : ''}`
  nodes.push(makeNode('attn', `${attnLabel} Attention`, attnSub, cx, y, '#14b8a6'))
  edges.push(makeEdge(prevInBlock, 'attn'))
  y += innerGap
  prevInBlock = 'attn'

  // MoE Router (if enabled)
  if (cfg.use_moe) {
    nodes.push(makeNode('moe', 'MoE Router', `experts=${cfg.num_experts} top-${cfg.num_experts_per_token}`, cx, y, '#ec4899'))
    edges.push(makeEdge(prevInBlock, 'moe'))
    y += innerGap
    prevInBlock = 'moe'
  }

  // Norm2
  nodes.push(makeNode('norm2', 'LayerNorm', cfg.norm_type, cx, y, '#f59e0b'))
  edges.push(makeEdge(prevInBlock, 'norm2'))
  y += innerGap
  prevInBlock = 'norm2'

  // FFN
  const dff = cfg.d_ff ? cfg.d_ff : Math.round(cfg.d_model * 2.67)
  nodes.push(makeNode('ffn', 'Feed Forward', `${cfg.activation} d_ff=${dff}`, cx, y, '#f97316'))
  edges.push(makeEdge(prevInBlock, 'ffn'))
  y += innerGap

  const blockBottomY = y - blockPad
  const blockHeight = blockBottomY - blockTopY + 60

  // Group node (visual box)
  nodes.push({
    id: 'layer-group',
    type: 'group',
    position: { x: cx - blockPad, y: blockTopY - blockPad },
    data: {},
    style: {
      width: 240 + blockPad * 2,
      height: blockHeight,
      backgroundColor: 'rgba(34,197,94,0.04)',
      border: '1.5px dashed rgba(34,197,94,0.3)',
      borderRadius: 10,
      pointerEvents: 'none',
    },
    draggable: false,
    selectable: false,
    connectable: false,
    zIndex: -1,
  })

  // Layer label overlay
  nodes.push({
    id: 'layer-label',
    type: 'default',
    position: { x: cx + 160, y: blockTopY },
    data: { label: `× ${cfg.n_layers}` },
    style: {
      background: 'transparent',
      border: 'none',
      color: '#22c55e',
      fontWeight: 700,
      fontSize: 14,
      padding: 0,
      boxShadow: 'none',
    },
    draggable: false,
    selectable: false,
    connectable: false,
  })

  // Final Norm
  nodes.push(makeNode('final-norm', 'Final Norm', cfg.norm_type, cx, y, '#f59e0b'))
  edges.push(makeEdge('ffn', 'final-norm'))
  y += gap

  // LM Head
  nodes.push(makeNode('lm-head', 'LM Head', `vocab=${cfg.vocab_size.toLocaleString()}`, cx, y, '#22c55e'))
  edges.push(makeEdge('final-norm', 'lm-head', '#22c55e'))

  return { nodes, edges }
}

// ─── Mamba2 graph ─────────────────────────────────────────────────────────────

function buildMamba2Graph(cfg: Mamba2Config): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = []
  const edges: Edge[] = []

  const cx = 100
  let y = 0
  const gap = 90

  // Token Embedding
  nodes.push(makeNode('embed', 'Token Embedding', `vocab=${cfg.vocab_size.toLocaleString()}`, cx, y, '#3b82f6'))
  y += gap

  const blockTopY = y
  const blockPad = 12
  const innerGap = 80

  // SSM Core
  nodes.push(makeNode('ssm', 'SSM Core', `state=${cfg.state_size} expand=${cfg.expand_factor}`, cx, y, '#06b6d4'))
  edges.push(makeEdge('embed', 'ssm'))
  y += innerGap

  // Gating
  nodes.push(makeNode('gate', 'Gating', `headdim=${cfg.headdim} chunk=${cfg.chunk_size}`, cx, y, '#8b5cf6'))
  edges.push(makeEdge('ssm', 'gate'))
  y += innerGap

  const blockHeight = y - blockTopY + 30

  nodes.push({
    id: 'layer-group',
    type: 'group',
    position: { x: cx - blockPad, y: blockTopY - blockPad },
    data: {},
    style: {
      width: 240 + blockPad * 2,
      height: blockHeight,
      backgroundColor: 'rgba(34,197,94,0.04)',
      border: '1.5px dashed rgba(34,197,94,0.3)',
      borderRadius: 10,
      pointerEvents: 'none',
    },
    draggable: false,
    selectable: false,
    connectable: false,
    zIndex: -1,
  })

  nodes.push({
    id: 'layer-label',
    type: 'default',
    position: { x: cx + 160, y: blockTopY },
    data: { label: `× ${cfg.n_layers}` },
    style: {
      background: 'transparent',
      border: 'none',
      color: '#22c55e',
      fontWeight: 700,
      fontSize: 14,
      padding: 0,
      boxShadow: 'none',
    },
    draggable: false,
    selectable: false,
    connectable: false,
  })

  // Final Norm
  nodes.push(makeNode('final-norm', 'Final Norm', cfg.norm_type, cx, y, '#f59e0b'))
  edges.push(makeEdge('gate', 'final-norm'))
  y += gap

  // LM Head
  nodes.push(makeNode('lm-head', 'LM Head', `vocab=${cfg.vocab_size.toLocaleString()}`, cx, y, '#22c55e'))
  edges.push(makeEdge('final-norm', 'lm-head', '#22c55e'))

  return { nodes, edges }
}

// ─── Custom node renderer ─────────────────────────────────────────────────────

function DiagramNode({ data }: { data: { label: string; sublabel: string; borderColor: string; bgColor: string } }) {
  return (
    <div
      style={{
        background: data.bgColor || '#1e293b',
        borderLeft: `3px solid ${data.borderColor}`,
        border: `1px solid #334155`,
        borderLeftColor: data.borderColor,
        borderRadius: 8,
        padding: '8px 14px',
        minWidth: 200,
        maxWidth: 240,
      }}
    >
      <div style={{ color: '#e2e8f0', fontSize: 12, fontWeight: 600 }}>{data.label}</div>
      {data.sublabel && (
        <div style={{ color: '#64748b', fontSize: 10, marginTop: 2 }}>{data.sublabel}</div>
      )}
    </div>
  )
}

const nodeTypes = { diagramNode: DiagramNode }

// ─── Main component ───────────────────────────────────────────────────────────

interface Props {
  config: ModelConfig
}

export default function ModelDiagram({ config }: Props) {
  const { nodes, edges } = useMemo(() => {
    if (config.architecture === 'transformer') {
      return buildTransformerGraph(config)
    }
    return buildMamba2Graph(config as Mamba2Config)
  }, [config])

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={true}
        zoomOnScroll={true}
      >
        <Background variant={BackgroundVariant.Dots} color="#1e293b" gap={24} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
