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
  ReactFlowInstance,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';

import ModelNode from './nodes/ModelNode';
import DatasetNode from './nodes/DatasetNode';
import AdamWNode from './nodes/AdamWNode';
import MuonNode from './nodes/MuonNode';
import LionNode from './nodes/LionNode';
import SophiaNode from './nodes/SophiaNode';
import CosineSchedulerNode from './nodes/CosineSchedulerNode';
import LinearSchedulerNode from './nodes/LinearSchedulerNode';
import PolynomialSchedulerNode from './nodes/PolynomialSchedulerNode';
import ConstantSchedulerNode from './nodes/ConstantSchedulerNode';
import HyperparametersNode from './nodes/HyperparametersNode';
import ConfigPanel from './ConfigPanel';
import ValidationPanel from './ValidationPanel';
import { Play } from 'lucide-react';
import { motion } from 'framer-motion';
import { useTraining } from '../context/TrainingContext';
import { startTraining, ModelConfigData, TrainingConfigData } from '../services/trainingApi';

const nodeTypes = {
  model: ModelNode,
  dataset: DatasetNode,
  adamw: AdamWNode,
  muon: MuonNode,
  lion: LionNode,
  sophia: SophiaNode,
  cosine: CosineSchedulerNode,
  linear: LinearSchedulerNode,
  polynomial: PolynomialSchedulerNode,
  constant: ConstantSchedulerNode,
  hyperparams: HyperparametersNode,
};

const initialNodes: Node[] = [
  {
    id: 'model-1',
    type: 'model',
    position: { x: 400, y: 300 },
    data: {
      label: 'Model',
      config_source: 'current',
      resume_training: false,
    },
  },
];

interface TrainingCanvasProps {
  onStartTraining: () => void;
  modelNodes: Node[];
  modelEdges: Edge[];
  trainingNodes: Node[];
  trainingEdges: Edge[];
  setTrainingNodes: (nodes: Node[]) => void;
  setTrainingEdges: (edges: Edge[]) => void;
}

const TrainingCanvas = ({
  onStartTraining,
  modelNodes,
  modelEdges,
  trainingNodes,
  trainingEdges,
  setTrainingNodes,
  setTrainingEdges
}: TrainingCanvasProps) => {
  const { updateTrainingState } = useTraining();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // Use parent state for nodes/edges to persist across tab switches
  const [nodes, setNodes, onNodesChange] = useNodesState(trainingNodes.length > 0 ? trainingNodes : initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(trainingEdges);

  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Sync local state changes back to parent
  useEffect(() => {
    setTrainingNodes(nodes);
  }, [nodes, setTrainingNodes]);

  useEffect(() => {
    setTrainingEdges(edges);
  }, [edges, setTrainingEdges]);

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

  const onConnect = useCallback(
    (params: Connection) => {
      if (!params.source || !params.target) return;

      const newEdge: Edge = {
        id: `e${params.source}-${params.target}`,
        source: params.source,
        target: params.target,
      };

      setEdges((eds) => [...eds, newEdge]);
    },
    [setEdges]
  );

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const onConfigUpdate = useCallback((nodeId: string, newData: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: newData }
          : node
      )
    );
    setSelectedNode((current) =>
      current?.id === nodeId ? { ...current, data: newData } : current
    );
  }, [setNodes]);

  const handleStartTraining = async () => {
    try {
      // Check if model is configured
      if (!modelNodes || !modelEdges) {
        alert('Please design your model in the Model Architecture tab first');
        return;
      }

      // Gather training config from nodes
      const datasetNode = nodes.find(n => n.type === 'dataset');
      const optimizerNode = nodes.find(n => ['adamw', 'muon', 'lion', 'sophia'].includes(n.type || ''));
      const schedulerNode = nodes.find(n => ['cosine', 'linear', 'polynomial', 'constant'].includes(n.type || ''));
      const hyperparamsNode = nodes.find(n => n.type === 'hyperparams');

      if (!datasetNode) {
        alert('Please add a Dataset node to the training canvas');
        return;
      }
      if (!optimizerNode) {
        alert('Please add an Optimizer node to the training canvas');
        return;
      }

      // Build training config
      const trainingConfig: TrainingConfigData = {
        datasets: [{
          name: datasetNode.data.dataset_name || 'HuggingFaceFW/fineweb-edu',
          subset: datasetNode.data.subset,
          split: datasetNode.data.split || 'train',
          weight: 1.0
        }],
        optimizer: optimizerNode.type || 'adamw',
        lr: optimizerNode.data.lr || hyperparamsNode?.data.lr || 0.001,
        weight_decay: optimizerNode.data.weight_decay || hyperparamsNode?.data.weight_decay || 0.01,
        batch_size: hyperparamsNode?.data.batch_size || 4,
        gradient_accumulation_steps: hyperparamsNode?.data.gradient_accumulation_steps || 1,
        max_steps: hyperparamsNode?.data.max_steps || 10000,
        warmup_steps: hyperparamsNode?.data.warmup_steps || schedulerNode?.data.warmup_steps || 100,
        scheduler: schedulerNode?.type || 'cosine',
        grad_clip: hyperparamsNode?.data.grad_clip || 1.0,
        eval_every: hyperparamsNode?.data.eval_every || 500,
        eval_steps: hyperparamsNode?.data.eval_steps || 100,
        // AdamW-specific
        adamw_beta1: optimizerNode.data.beta1 || 0.9,
        adamw_beta2: optimizerNode.data.beta2 || 0.999,
        adamw_eps: optimizerNode.data.eps || 1e-8,
        // Muon-specific
        muon_momentum: optimizerNode.data.momentum || 0.95,
        muon_nesterov: optimizerNode.data.nesterov ?? true,
        // Lion-specific
        lion_beta1: optimizerNode.data.beta1 || 0.9,
        lion_beta2: optimizerNode.data.beta2 || 0.99,
        // Sophia-specific
        sophia_beta1: optimizerNode.data.beta1 || 0.965,
        sophia_beta2: optimizerNode.data.beta2 || 0.99,
        sophia_rho: optimizerNode.data.rho || 0.04,
      };

      // Build model config from Model Architecture nodes
      const tokenizerNode = modelNodes.find(n => n.type === 'tokenizer');
      const embeddingNode = modelNodes.find(n => n.type === 'embedding');
      const posEncodingNode = modelNodes.find(n => ['rope', 'alibi', 'yarn', 'sinusoidal'].includes(n.type || ''));
      const attentionNode = modelNodes.find(n => ['mha', 'gqa', 'mqa', 'mla'].includes(n.type || ''));
      const normNode = modelNodes.find(n => ['rmsnorm', 'layernorm'].includes(n.type || ''));
      const ffnNode = modelNodes.find(n => ['swiglu', 'gelu', 'relu'].includes(n.type || ''));
      const lmheadNode = modelNodes.find(n => n.type === 'lmhead');

      // Find loop edge for n_layers
      const loopEdge = modelEdges.find(e => e.data?.isLoop);

      const modelConfig: ModelConfigData = {
        model_architecture: 'transformer',
        tokenizer_name: tokenizerNode?.data.tokenizer_name || 'Qwen/Qwen2.5-0.5B',
        d_model: embeddingNode?.data.d_model || 896,
        n_layers: loopEdge?.data?.repeatCount || 24,
        vocab_size: embeddingNode?.data.vocab_size || 151936,
        max_seq_len: posEncodingNode?.data.max_seq_len || 1024,
        positional_encoding: posEncodingNode?.type || 'rope',
        attention_type: attentionNode?.type || 'gqa',
        activation: ffnNode?.type || 'swiglu',
        n_heads: attentionNode?.data.n_heads || 14,
        n_kv_heads: attentionNode?.data.n_kv_heads || 2,
        d_ff: ffnNode?.data.d_ff || 3584,
        norm_type: normNode?.type || 'rmsnorm',
        norm_eps: 1e-6,
        dropout: 0.0,
        tie_word_embeddings: lmheadNode?.data.tie_weights ?? true,
        model_type: 'custom_transformer',
        // MLA-specific parameters (only if MLA attention is used)
        d_latent: attentionNode?.data.d_latent || undefined,
        d_rope_latent: attentionNode?.data.d_rope_latent || undefined,
      };

      // Reset training state
      updateTrainingState({
        isTraining: true,
        isPaused: false,
        progress: 0,
        currentStep: 0,
        maxSteps: trainingConfig.max_steps,
        currentLoss: null,
        currentPPL: null,
        currentLR: null,
      });

      // Start training via API
      await startTraining(modelConfig, trainingConfig);

      // Navigate to monitor
      onStartTraining();
    } catch (error: any) {
      console.error('Failed to start training:', error);
      alert(`Failed to start training: ${error.message}`);
      updateTrainingState({
        isTraining: false,
        isPaused: false,
      });
    }
  };

  return (
    <div className="absolute inset-0" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        onInit={setReactFlowInstance}
        onDrop={onDrop}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        fitView
        className="bg-slate-900"
      >
        <Background color="#475569" variant={BackgroundVariant.Dots} gap={16} size={1} />
        <Controls className="bg-slate-800 border border-slate-700" />
        <MiniMap
          className="bg-slate-800 border border-slate-700"
          nodeColor={(node) => {
            switch (node.type) {
              case 'dataset':
                return '#3b82f6';
              case 'model':
                return '#6366f1';
              case 'adamw':
                return '#ef4444';
              case 'muon':
                return '#f97316';
              case 'lion':
                return '#f59e0b';
              case 'sophia':
                return '#f43f5e';
              case 'cosine':
                return '#a855f7';
              case 'linear':
                return '#8b5cf6';
              case 'polynomial':
                return '#d946ef';
              case 'constant':
                return '#64748b';
              case 'hyperparams':
                return '#22c55e';
              default:
                return '#64748b';
            }
          }}
        />
      </ReactFlow>

      {/* Stats & Validation Panel */}
      <div className={`absolute top-4 right-4 space-y-4 transition-all duration-300 ${selectedNode ? 'mr-96' : ''}`}>
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 shadow-xl">
          <h3 className="text-white font-semibold text-sm mb-3">Training Stats</h3>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">Datasets:</span>
              <span className="text-white font-mono">{nodes.filter(n => n.type === 'dataset').length}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">Optimizers:</span>
              <span className="text-white font-mono">{nodes.filter(n => ['adamw', 'muon', 'lion', 'sophia'].includes(n.type || '')).length}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-400">Schedulers:</span>
              <span className="text-white font-mono">{nodes.filter(n => ['cosine', 'linear', 'polynomial', 'constant'].includes(n.type || '')).length}</span>
            </div>
          </div>
        </div>
        <ValidationPanel nodes={nodes} edges={edges} mode="training" />
      </div>

      {/* Configuration Panel */}
      <ConfigPanel
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onUpdate={onConfigUpdate}
      />

      {/* Start Training Button */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
      >
        <button
          onClick={handleStartTraining}
          className="px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white rounded-xl font-bold text-lg flex items-center gap-3 shadow-2xl transition-all hover:scale-105 border-2 border-green-400"
        >
          <Play className="w-6 h-6" />
          Start Training
        </button>
      </motion.div>
    </div>
  );
};

const TrainingCanvasWrapper = (props: TrainingCanvasProps) => (
  <ReactFlowProvider>
    <TrainingCanvas {...props} />
  </ReactFlowProvider>
);

export default TrainingCanvasWrapper;
