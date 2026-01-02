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

import BaseModelNode from './nodes/BaseModelNode';
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
import LoRANode from './nodes/LoRANode';
import ConfigPanel from './ConfigPanel';
import ValidationPanel from './ValidationPanel';
import { Play } from 'lucide-react';
import { motion } from 'framer-motion';
import { useTraining } from '../context/TrainingContext';
import { startSFT, SFTConfigData } from '../services/trainingApi';

const nodeTypes = {
  basemodel: BaseModelNode,
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
  lora: LoRANode,
};

const initialNodes: Node[] = [
  {
    id: 'basemodel-1',
    type: 'basemodel',
    position: { x: 400, y: 300 },
    data: {
      label: 'Base Model',
      checkpoint_path: '/app/data/checkpoints/best_model.pt',
    },
  },
];

interface SFTCanvasProps {
  onStartTraining: () => void;
  sftNodes: Node[];
  sftEdges: Edge[];
  setSftNodes: (nodes: Node[]) => void;
  setSftEdges: (edges: Edge[]) => void;
}

const SFTCanvas = ({
  onStartTraining,
  sftNodes,
  sftEdges,
  setSftNodes,
  setSftEdges
}: SFTCanvasProps) => {
  const { updateTrainingState, clearMetrics } = useTraining();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // Use parent state for nodes/edges to persist across tab switches
  const [nodes, setNodes, onNodesChange] = useNodesState(sftNodes.length > 0 ? sftNodes : initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(sftEdges.length > 0 ? sftEdges : []);

  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Sync local state changes back to parent
  useEffect(() => {
    setSftNodes(nodes);
  }, [nodes, setSftNodes]);

  useEffect(() => {
    setSftEdges(edges);
  }, [edges, setSftEdges]);

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

  const handleStartSFT = async () => {
    try {
      // Gather SFT config from nodes
      const baseModelNode = nodes.find(n => n.type === 'basemodel');
      const datasetNodes = nodes.filter(n => n.type === 'dataset');
      const optimizerNode = nodes.find(n => ['adamw', 'muon', 'lion', 'sophia'].includes(n.type || ''));
      const schedulerNode = nodes.find(n => ['cosine', 'linear', 'polynomial', 'constant'].includes(n.type || ''));
      const hyperparamsNode = nodes.find(n => n.type === 'hyperparams');
      const loraNode = nodes.find(n => n.type === 'lora');

      if (!baseModelNode) {
        alert('Please add a Base Model node to specify which checkpoint to fine-tune');
        return;
      }
      if (datasetNodes.length === 0) {
        alert('Please add at least one Dataset node for SFT');
        return;
      }
      if (!optimizerNode) {
        alert('Please add an Optimizer node');
        return;
      }

      // Build datasets config with weights
      const datasets = datasetNodes.map(node => ({
        name: node.data.dataset_name || 'HuggingFaceH4/ultrachat_200k',
        subset: node.data.subset,
        split: node.data.split || 'train_sft',
        weight: node.data.weight || 1.0
      }));

      // Build SFT config
      const sftConfig: SFTConfigData = {
        policy_checkpoint: baseModelNode.data.checkpoint_path || '/app/data/checkpoints/best_model.pt',
        datasets: datasets,
        optimizer: optimizerNode.type || 'adamw',
        lr: optimizerNode.data.lr || hyperparamsNode?.data.lr || 5e-6,
        weight_decay: optimizerNode.data.weight_decay || hyperparamsNode?.data.weight_decay || 0.01,
        batch_size: hyperparamsNode?.data.batch_size || 4,
        gradient_accumulation_steps: hyperparamsNode?.data.gradient_accumulation_steps || 16,
        max_steps: hyperparamsNode?.data.max_steps || 5000,
        warmup_steps: hyperparamsNode?.data.warmup_steps || schedulerNode?.data.warmup_steps || 100,
        scheduler: schedulerNode?.type || 'cosine',
        max_grad_norm: hyperparamsNode?.data.grad_clip || 1.0,
        log_every: 10,
        save_every: 500,
        eval_every: hyperparamsNode?.data.eval_every || 500,
        eval_steps: hyperparamsNode?.data.eval_steps || 50,
        save_best_only: true,
        output_dir: '/app/data/sft_checkpoints',
        // LoRA configuration (optional - enabled if node exists)
        use_lora: loraNode ? true : false,
        lora_preset: loraNode?.data.preset || 'minimal',
        lora_target_modules: loraNode?.data.lora_target_modules,
        lora_r: loraNode?.data.lora_r || 8,
        lora_alpha: loraNode?.data.lora_alpha || 16,
        lora_dropout: loraNode?.data.lora_dropout || 0.05,
        // Optimizer-specific parameters
        adamw_beta1: optimizerNode.data.beta1 || 0.9,
        adamw_beta2: optimizerNode.data.beta2 || 0.999,
        adamw_eps: optimizerNode.data.eps || 1e-8,
        muon_momentum: optimizerNode.data.momentum || 0.95,
        muon_nesterov: optimizerNode.data.nesterov ?? true,
        lion_beta1: optimizerNode.data.beta1 || 0.9,
        lion_beta2: optimizerNode.data.beta2 || 0.99,
        sophia_beta1: optimizerNode.data.beta1 || 0.965,
        sophia_beta2: optimizerNode.data.beta2 || 0.99,
        sophia_rho: optimizerNode.data.rho || 0.04,
      };

      // Clear previous metrics and logs
      clearMetrics();

      // Reset training state
      updateTrainingState({
        isTraining: true,
        isPaused: false,
        progress: 0,
        currentStep: 0,
        maxSteps: sftConfig.max_steps,
        currentLoss: null,
        currentPPL: null,
        currentLR: null,
      });

      // Start SFT via API
      await startSFT(sftConfig);

      // Navigate to monitor
      onStartTraining();
    } catch (error: any) {
      console.error('Failed to start SFT:', error);
      alert(`Failed to start SFT: ${error.message}`);
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
              case 'basemodel':
                return '#3b82f6'; // blue
              case 'dataset':
                return '#06b6d4'; // cyan
              case 'adamw':
                return '#ef4444'; // red
              case 'muon':
                return '#f97316'; // orange
              case 'lion':
                return '#f59e0b'; // amber
              case 'sophia':
                return '#f43f5e'; // rose
              case 'cosine':
                return '#a855f7'; // purple
              case 'linear':
                return '#8b5cf6'; // violet
              case 'polynomial':
                return '#d946ef'; // fuchsia
              case 'constant':
                return '#64748b'; // slate
              case 'hyperparams':
                return '#22c55e'; // green
              case 'lora':
                return '#a855f7'; // purple
              default:
                return '#64748b'; // slate
            }
          }}
        />
      </ReactFlow>

      {/* Validation Panel */}
      <div className={`absolute top-4 right-4 space-y-4 transition-all duration-300 ${selectedNode ? 'mr-96' : ''}`}>
        <ValidationPanel nodes={nodes} edges={edges} mode="sft" />
      </div>

      {/* Configuration Panel */}
      <ConfigPanel
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onUpdate={onConfigUpdate}
      />

      {/* Start SFT Button */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
      >
        <button
          onClick={handleStartSFT}
          className="px-8 py-4 bg-gradient-to-r from-purple-600 to-violet-600 hover:from-purple-700 hover:to-violet-700 text-white rounded-xl font-bold text-lg flex items-center gap-3 shadow-2xl transition-all hover:scale-105 border-2 border-purple-400"
        >
          <Play className="w-6 h-6" />
          Start SFT
        </button>
      </motion.div>
    </div>
  );
};

const SFTCanvasWrapper = (props: SFTCanvasProps) => (
  <ReactFlowProvider>
    <SFTCanvas {...props} />
  </ReactFlowProvider>
);

export default SFTCanvasWrapper;
