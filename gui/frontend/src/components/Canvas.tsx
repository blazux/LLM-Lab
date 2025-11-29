import { useCallback, useRef, useState, useEffect } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  Connection,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
  ReactFlowProvider,
  ReactFlowInstance,
} from 'reactflow';
import 'reactflow/dist/style.css';

import Sidebar from './Sidebar';
import TrainingCanvas from './TrainingCanvas';
import TrainingMonitor from './TrainingMonitor';
import TrainingStatusWidget from './TrainingStatusWidget';
import Inference from './Inference';
import { useTraining } from '../context/TrainingContext';
import { AnimatePresence } from 'framer-motion';
import TokenizerNode from './nodes/TokenizerNode';
import EmbeddingNode from './nodes/EmbeddingNode';
import LayerNode from './nodes/LayerNode';
import LMHeadNode from './nodes/LMHeadNode';
import ConfigPanel from './ConfigPanel';
import LoopEdge from './edges/LoopEdge';
import EdgeConfigPanel from './EdgeConfigPanel';
import ValidationPanel from './ValidationPanel';
// Positional Encoding nodes
import RoPENode from './nodes/RoPENode';
import ALiBiNode from './nodes/ALiBiNode';
import YARNNode from './nodes/YARNNode';
import SinusoidalNode from './nodes/SinusoidalNode';
// Attention nodes
import MHANode from './nodes/MHANode';
import GQANode from './nodes/GQANode';
import MQANode from './nodes/MQANode';
import MLANode from './nodes/MLANode';
// Normalization nodes
import RMSNormNode from './nodes/RMSNormNode';
import LayerNormNode from './nodes/LayerNormNode';
// FFN nodes
import SwiGLUNode from './nodes/SwiGLUNode';
import GELUNode from './nodes/GELUNode';
import ReLUNode from './nodes/ReLUNode';
import { generateConfigFromNodes, downloadConfig } from '../utils/configGenerator';

const nodeTypes = {
  // Core components
  tokenizer: TokenizerNode,
  embedding: EmbeddingNode,
  layer: LayerNode,
  lmhead: LMHeadNode,
  // Positional encoding
  rope: RoPENode,
  alibi: ALiBiNode,
  yarn: YARNNode,
  sinusoidal: SinusoidalNode,
  // Attention mechanisms
  mha: MHANode,
  gqa: GQANode,
  mqa: MQANode,
  mla: MLANode,
  // Normalization
  rmsnorm: RMSNormNode,
  layernorm: LayerNormNode,
  // Feed forward
  swiglu: SwiGLUNode,
  gelu: GELUNode,
  relu: ReLUNode,
};

const edgeTypes = {
  loop: LoopEdge,
};

const initialNodes: Node[] = [
  {
    id: '1',
    type: 'tokenizer',
    position: { x: 250, y: 500 },
    data: { label: 'Tokenizer', tokenizer_name: 'Qwen/Qwen2.5-0.5B' },
  },
];

const Canvas = () => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [activeTab, setActiveTab] = useState<'model' | 'training' | 'monitor' | 'inference'>('model');

  // Model Architecture state
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Training Canvas state (separate from model architecture)
  const [trainingNodes, setTrainingNodes] = useState<Node[]>([]);
  const [trainingEdges, setTrainingEdges] = useState<Edge[]>([]);

  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<Edge | null>(null);
  const { trainingState } = useTraining();

  const handleGenerateConfig = useCallback(() => {
    const config = generateConfigFromNodes(nodes, edges);
    downloadConfig(config);
  }, [nodes, edges]);

  const handleLoadModelPreset = useCallback((presetNodes: Node[], presetEdges: Edge[]) => {
    setNodes(presetNodes);
    setEdges(presetEdges);
    setSelectedNode(null);
    setSelectedEdge(null);
  }, [setNodes, setEdges]);

  const handleLoadTrainingPreset = useCallback((presetNodes: Node[], presetEdges: Edge[]) => {
    setTrainingNodes(presetNodes);
    setTrainingEdges(presetEdges);
  }, [setTrainingNodes, setTrainingEdges]);

  const handleClearModelCanvas = useCallback(() => {
    if (confirm('Are you sure you want to clear the model canvas?')) {
      setNodes([]);
      setEdges([]);
      setSelectedNode(null);
      setSelectedEdge(null);
    }
  }, [setNodes, setEdges]);

  const handleClearTrainingCanvas = useCallback(() => {
    if (confirm('Are you sure you want to clear the training canvas?')) {
      setTrainingNodes([]);
      setTrainingEdges([]);
    }
  }, []);

  // Calculate model parameters (matches src/config/config.py count_params())
  const calculateParameters = useCallback(() => {
    const embeddingNode = nodes.find(n => n.type === 'embedding');
    const attentionNode = nodes.find(n => ['mha', 'gqa', 'mqa', 'mla'].includes(n.type || ''));
    const ffnNode = nodes.find(n => ['swiglu', 'gelu', 'relu'].includes(n.type || ''));
    const loopEdge = edges.find(e => e.data?.isLoop);

    if (!embeddingNode) return null;

    const d_model = embeddingNode.data.d_model || 896;
    const vocab_size = embeddingNode.data.vocab_size || 151936;
    const n_layers = loopEdge?.data?.repeatCount || 24;
    const d_ff = ffnNode?.data.d_ff || (d_model * 4);
    const attention_type = attentionNode?.type || 'gqa';

    // Embeddings
    const embed_params = vocab_size * d_model;

    // Calculate d_k (head dimension)
    const n_heads = attentionNode?.data.n_heads || 14;
    const d_k = d_model / n_heads;

    // Per-layer attention (exact formulas from Python)
    let attn_params: number;
    if (attention_type === 'mha') {
      // Q, K, V, O projections: all are d_model x d_model
      attn_params = 4 * d_model * d_model;
    } else if (attention_type === 'mqa') {
      // Q projection: d_model x d_model
      // Single K,V: 2 * d_model * d_k
      // O projection: d_model x d_model
      attn_params = d_model * d_model + 2 * d_model * d_k + d_model * d_model;
    } else if (attention_type === 'mla') {
      // MLA uses latent dimensions
      const d_latent = attentionNode?.data.d_latent || Math.max(d_model / 4, d_k);
      const q_params = d_model * d_model;
      const kv_down_params = d_model * d_latent;
      const k_up_params = d_latent * d_model;
      const v_up_params = d_latent * d_model;
      const out_params = d_model * d_model;
      attn_params = q_params + kv_down_params + k_up_params + v_up_params + out_params;
    } else { // gqa
      // Q projection: d_model x d_model
      // K,V projections: 2 * n_kv_heads * d_k * d_model
      // O projection: d_model x d_model
      const n_kv_heads = attentionNode?.data.n_kv_heads || 2;
      attn_params = d_model * d_model + 2 * n_kv_heads * d_k * d_model + d_model * d_model;
    }

    // Per-layer FFN
    const activation = ffnNode?.type || 'swiglu';
    const ff_params = activation === 'swiglu'
      ? 3 * d_model * d_ff  // SwiGLU has 3 projections
      : 2 * d_model * d_ff; // Others have 2 projections

    // Per-layer normalization (2 per layer: pre-attn and pre-ffn)
    const norm_params = 4 * d_model;

    const layer_params = attn_params + ff_params + norm_params;
    const total_params = embed_params + (n_layers * layer_params) + d_model; // +d_model for final norm

    return total_params;
  }, [nodes, edges]);

  const formatParams = (params: number | null) => {
    if (params === null) return 'Configure model';
    if (params >= 1e9) return `${(params / 1e9).toFixed(2)}B`;
    if (params >= 1e6) return `${(params / 1e6).toFixed(0)}M`;
    if (params >= 1e3) return `${(params / 1e3).toFixed(0)}K`;
    return params.toString();
  };

  // Auto-sync vocab_size from embedding to lmhead
  useEffect(() => {
    const embeddingNode = nodes.find(n => n.type === 'embedding');
    const lmheadNode = nodes.find(n => n.type === 'lmhead');

    if (embeddingNode && lmheadNode && embeddingNode.data.vocab_size) {
      // Only update if lmhead vocab_size is different or not set
      if (lmheadNode.data.vocab_size !== embeddingNode.data.vocab_size) {
        setNodes((nds) =>
          nds.map((node) =>
            node.type === 'lmhead'
              ? { ...node, data: { ...node.data, vocab_size: embeddingNode.data.vocab_size } }
              : node
          )
        );
      }
    }
  }, [nodes, setNodes]);

  // Detect if edge creates a loop
  const isLoopEdge = useCallback((sourceNodeId: string, targetNodeId: string, currentNodes: Node[]) => {
    const sourceNode = currentNodes.find(n => n.id === sourceNodeId);
    const targetNode = currentNodes.find(n => n.id === targetNodeId);

    if (!sourceNode || !targetNode) return false;

    // Check if target is "below" source (loop back)
    // Since we build bottom-to-top, data flows UP, so loop goes back DOWN
    return targetNode.position.y > sourceNode.position.y;
  }, []);

  const onConnect = useCallback(
    (params: Connection) => {
      if (!params.source || !params.target) return;

      const isLoop = isLoopEdge(params.source, params.target, nodes);

      const newEdge: Edge = {
        id: `e${params.source}-${params.target}`,
        source: params.source,
        target: params.target,
        type: isLoop ? 'loop' : 'default',
        data: {
          isLoop,
          repeatCount: isLoop ? 24 : undefined,
        },
      };

      setEdges((eds) => [...eds, newEdge]);
    },
    [setEdges, nodes, isLoopEdge]
  );

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
    setSelectedEdge(null);
  }, []);

  const onEdgeClick = useCallback((_event: React.MouseEvent, edge: Edge) => {
    if (edge.data?.isLoop) {
      setSelectedEdge(edge);
      setSelectedNode(null);
    }
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setSelectedEdge(null);
  }, []);

  const onNodesDelete = useCallback((deleted: Node[]) => {
    // Close config panel if the deleted node was selected
    deleted.forEach((node) => {
      if (selectedNode?.id === node.id) {
        setSelectedNode(null);
      }
    });
  }, [selectedNode]);

  const onEdgesDelete = useCallback((deleted: Edge[]) => {
    // Close edge config panel if the deleted edge was selected
    deleted.forEach((edge) => {
      if (selectedEdge?.id === edge.id) {
        setSelectedEdge(null);
      }
    });
  }, [selectedEdge]);

  const onConfigUpdate = useCallback((nodeId: string, newData: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: newData }
          : node
      )
    );
    // Update selected node to reflect changes
    setSelectedNode((current) =>
      current?.id === nodeId ? { ...current, data: newData } : current
    );
  }, [setNodes]);

  const onEdgeUpdate = useCallback((edgeId: string, newData: any) => {
    setEdges((eds) =>
      eds.map((edge) =>
        edge.id === edgeId
          ? { ...edge, data: newData }
          : edge
      )
    );
    // Update selected edge to reflect changes
    setSelectedEdge((current) =>
      current?.id === edgeId ? { ...current, data: newData } : current
    );
  }, [setEdges]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      const label = event.dataTransfer.getData('label');

      if (typeof type === 'undefined' || !type || !reactFlowInstance) {
        return;
      }

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type,
        position,
        data: { label },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  return (
    <>
      <Sidebar
        nodes={activeTab === 'model' ? nodes : trainingNodes}
        onGenerateConfig={handleGenerateConfig}
        activeTab={activeTab}
        onLoadPreset={activeTab === 'model' ? handleLoadModelPreset : handleLoadTrainingPreset}
        onClearCanvas={activeTab === 'model' ? handleClearModelCanvas : handleClearTrainingCanvas}
      />
      <div className="flex-1 relative flex flex-col" ref={reactFlowWrapper}>
        {/* Tab Switcher */}
        <div className="bg-slate-800 border-b border-slate-700 px-6 py-3 flex gap-2">
          <button
            onClick={() => setActiveTab('model')}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
              activeTab === 'model'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            🧠 Model Architecture
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
              activeTab === 'training'
                ? 'bg-green-600 text-white shadow-lg'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            ⚙️ Training Config
          </button>
          <button
            onClick={() => setActiveTab('monitor')}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
              activeTab === 'monitor'
                ? 'bg-orange-600 text-white shadow-lg'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            📊 Training Monitor
          </button>
          <button
            onClick={() => setActiveTab('inference')}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
              activeTab === 'inference'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            💬 Inference
          </button>
        </div>

        {/* Canvas Area */}
        <div className="flex-1 relative">
          {activeTab === 'model' && (
            <>
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={onNodeClick}
                onEdgeClick={onEdgeClick}
                onPaneClick={onPaneClick}
                onNodesDelete={onNodesDelete}
                onEdgesDelete={onEdgesDelete}
                onInit={setReactFlowInstance}
                onDrop={onDrop}
                onDragOver={onDragOver}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                fitView
                deleteKeyCode={['Backspace', 'Delete']}
                className="bg-slate-900"
              >
                <Background color="#475569" variant={BackgroundVariant.Dots} gap={16} size={1} />
                <Controls className="bg-slate-800 border border-slate-700" />
                <MiniMap
                  className="bg-slate-800 border border-slate-700"
                  nodeColor={(node) => {
                    switch (node.type) {
                      // Core components
                      case 'tokenizer':
                        return '#8b5cf6'; // purple
                      case 'embedding':
                        return '#14b8a6'; // teal
                      case 'layer':
                        return '#6366f1'; // indigo
                      case 'lmhead':
                        return '#ec4899'; // pink
                      // Positional encoding
                      case 'rope':
                        return '#3b82f6'; // blue
                      case 'alibi':
                        return '#0ea5e9'; // sky
                      case 'yarn':
                        return '#6366f1'; // indigo
                      case 'sinusoidal':
                        return '#8b5cf6'; // violet
                      // Attention mechanisms
                      case 'mha':
                        return '#10b981'; // green
                      case 'gqa':
                        return '#059669'; // emerald
                      case 'mqa':
                        return '#14b8a6'; // teal
                      case 'mla':
                        return '#06b6d4'; // cyan
                      // Normalization
                      case 'rmsnorm':
                        return '#eab308'; // yellow
                      case 'layernorm':
                        return '#f59e0b'; // amber
                      // Feed forward
                      case 'swiglu':
                        return '#f97316'; // orange
                      case 'gelu':
                        return '#ef4444'; // red
                      case 'relu':
                        return '#ec4899'; // pink
                      default:
                        return '#64748b'; // slate
                    }
                  }}
                />
              </ReactFlow>

              {/* Stats & Validation Panel */}
              <div className={`absolute top-4 right-4 space-y-4 transition-all duration-300 ${selectedNode ? 'mr-96' : ''}`}>
                <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 shadow-xl">
                  <h3 className="text-white font-semibold text-sm mb-3">Model Stats</h3>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-400">Blocks:</span>
                      <span className="text-white font-mono">{nodes.length}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-400">Connections:</span>
                      <span className="text-white font-mono">{edges.length}</span>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-400">Parameters:</span>
                      <span className="text-purple-400 font-mono">{formatParams(calculateParameters())}</span>
                    </div>
                  </div>
                </div>
                <ValidationPanel nodes={nodes} edges={edges} mode="model" />
              </div>

              {/* Configuration Panel */}
              <ConfigPanel
                node={selectedNode}
                onClose={() => setSelectedNode(null)}
                onUpdate={onConfigUpdate}
              />

              {/* Edge Configuration Panel */}
              <EdgeConfigPanel
                edge={selectedEdge}
                onClose={() => setSelectedEdge(null)}
                onUpdate={onEdgeUpdate}
              />
            </>
          )}

          {/* Training Canvas */}
          {activeTab === 'training' && (
            <TrainingCanvas
              onStartTraining={() => setActiveTab('monitor')}
              modelNodes={nodes}
              modelEdges={edges}
              trainingNodes={trainingNodes}
              trainingEdges={trainingEdges}
              setTrainingNodes={setTrainingNodes}
              setTrainingEdges={setTrainingEdges}
            />
          )}

          {/* Training Monitor */}
          {activeTab === 'monitor' && (
            <TrainingMonitor />
          )}

          {/* Inference */}
          {activeTab === 'inference' && (
            <Inference />
          )}
        </div>
      </div>

      {/* Floating Training Status Widget */}
      <AnimatePresence>
        {trainingState.isTraining && activeTab !== 'monitor' && (
          <TrainingStatusWidget
            progress={trainingState.progress}
            loss={trainingState.currentLoss}
            perplexity={trainingState.currentPPL}
            onNavigate={() => setActiveTab('monitor')}
          />
        )}
      </AnimatePresence>
    </>
  );
};

const CanvasWrapper = () => (
  <ReactFlowProvider>
    <Canvas />
  </ReactFlowProvider>
);

export default CanvasWrapper;
