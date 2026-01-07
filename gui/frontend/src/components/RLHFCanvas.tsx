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
import RLHFHyperparametersNode from './nodes/RLHFHyperparametersNode';
import LoRANode from './nodes/LoRANode';
import PPORewardModelNode from './nodes/PPORewardModelNode';
import DPOReferenceModelNode from './nodes/DPOReferenceModelNode';
import GRPORewardModelNode from './nodes/GRPORewardModelNode';
import ConfigPanel from './ConfigPanel';
import ValidationPanel from './ValidationPanel';
import { Play } from 'lucide-react';
import { motion } from 'framer-motion';
import { useTraining } from '../context/TrainingContext';
import { startRLHF, RLHFConfigData } from '../services/trainingApi';

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
  rlhf_hyperparams: RLHFHyperparametersNode,
  lora: LoRANode,
  ppo_reward: PPORewardModelNode,
  dpo_reference: DPOReferenceModelNode,
  grpo_reward: GRPORewardModelNode,
};

const initialNodes: Node[] = [
  {
    id: 'basemodel-1',
    type: 'basemodel',
    position: { x: 400, y: 300 },
    data: {
      label: 'Policy Model',
      checkpoint_path: '/app/data/best_model.pt',
    },
  },
];

interface RLHFCanvasProps {
  onStartTraining: () => void;
  rlhfNodes: Node[];
  rlhfEdges: Edge[];
  setRlhfNodes: (nodes: Node[]) => void;
  setRlhfEdges: (edges: Edge[]) => void;
}

const RLHFCanvas = ({
  onStartTraining,
  rlhfNodes,
  rlhfEdges,
  setRlhfNodes,
  setRlhfEdges
}: RLHFCanvasProps) => {
  const { updateTrainingState, clearMetrics } = useTraining();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const [nodes, setNodes, onNodesChange] = useNodesState(rlhfNodes.length > 0 ? rlhfNodes : initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(rlhfEdges.length > 0 ? rlhfEdges : []);

  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  // Sync local state changes back to parent
  useEffect(() => {
    setRlhfNodes(nodes);
  }, [nodes, setRlhfNodes]);

  useEffect(() => {
    setRlhfEdges(edges);
  }, [edges, setRlhfEdges]);

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
        sourceHandle: params.sourceHandle,
        targetHandle: params.targetHandle,
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

  const handleStartRLHF = async () => {
    try {
      // Detect algorithm based on which model node exists
      const ppoNode = nodes.find(n => n.type === 'ppo_reward');
      const dpoNode = nodes.find(n => n.type === 'dpo_reference');
      const grpoNode = nodes.find(n => n.type === 'grpo_reward');

      // Validate exactly one algorithm node
      const algorithmNodes = [ppoNode, dpoNode, grpoNode].filter(Boolean);
      if (algorithmNodes.length === 0) {
        alert('Please add a PPO, DPO, or GRPO model node to select an algorithm');
        return;
      }
      if (algorithmNodes.length > 1) {
        alert('Please use only one algorithm at a time (remove extra PPO/DPO/GRPO nodes)');
        return;
      }

      // Determine algorithm
      let algorithm: 'ppo' | 'dpo' | 'grpo';
      if (ppoNode) algorithm = 'ppo';
      else if (dpoNode) algorithm = 'dpo';
      else algorithm = 'grpo';

      // Gather common nodes
      const policyNode = nodes.find(n => n.type === 'basemodel');
      const datasetNodes = nodes.filter(n => n.type === 'dataset');
      const optimizerNode = nodes.find(n => ['adamw', 'muon', 'lion', 'sophia'].includes(n.type || ''));
      const hyperparamsNode = nodes.find(n => n.type === 'rlhf_hyperparams');
      const loraNode = nodes.find(n => n.type === 'lora');

      // Validate required nodes
      if (!policyNode) {
        alert('Please add a Policy Model node (Base Model)');
        return;
      }
      if (datasetNodes.length === 0) {
        alert('Please add at least one Dataset node');
        return;
      }
      if (!optimizerNode) {
        alert('Please add an Optimizer node');
        return;
      }

      // Build datasets config
      const datasets = datasetNodes.map(node => ({
        name: node.data.dataset_name || node.data.name || 'Anthropic/hh-rlhf',
        subset: node.data.subset,
        split: node.data.split || 'train',
        weight: node.data.weight || 1.0
      }));

      // Build base RLHF config
      const rlhfConfig: RLHFConfigData = {
        algorithm,
        policy_checkpoint: policyNode.data.checkpoint_path || '/app/data/best_model.pt',
        datasets: datasets,
        optimizer: optimizerNode.type || 'adamw',
        learning_rate: hyperparamsNode?.data.learning_rate || 1.4e-5,
        weight_decay: hyperparamsNode?.data.weight_decay || 0.0,
        batch_size: hyperparamsNode?.data.batch_size || 128,
        mini_batch_size: hyperparamsNode?.data.mini_batch_size || 32,
        gradient_accumulation_steps: hyperparamsNode?.data.gradient_accumulation_steps || 1,
        max_steps: hyperparamsNode?.data.max_steps || 10000,
        max_grad_norm: hyperparamsNode?.data.max_grad_norm || 1.0,
        log_every: 10,
        save_every: 500,
        eval_every: 500,
        output_dir: '/app/data',

        // Generation parameters
        max_new_tokens: hyperparamsNode?.data.max_new_tokens || 128,
        temperature: hyperparamsNode?.data.temperature || 1.0,
        top_k: hyperparamsNode?.data.top_k || 0,
        top_p: hyperparamsNode?.data.top_p || 1.0,

        // LoRA configuration
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

        // Algorithm-specific parameters
        reward_model_name: '',
        reference_checkpoint: '',
        ppo_epochs: 4,
        clip_range: 0.2,
        gamma: 1.0,
        gae_lambda: 0.95,
        vf_coef: 0.1,
        group_size: 4,
        grpo_temperature: 1.0,
      };

      // Add algorithm-specific configuration
      if (algorithm === 'ppo' && ppoNode) {
        rlhfConfig.reward_model_name = ppoNode.data.model_name || 'OpenAssistant/reward-model-deberta-v3-large-v2';
        rlhfConfig.ppo_epochs = hyperparamsNode?.data.ppo_epochs || 4;
        rlhfConfig.clip_range = hyperparamsNode?.data.clip_range || 0.2;
        rlhfConfig.gamma = hyperparamsNode?.data.gamma || 1.0;
        rlhfConfig.gae_lambda = hyperparamsNode?.data.gae_lambda || 0.95;
        rlhfConfig.vf_coef = hyperparamsNode?.data.vf_coef || 0.1;
      } else if (algorithm === 'dpo' && dpoNode) {
        rlhfConfig.reference_checkpoint = dpoNode.data.checkpoint_path || policyNode.data.checkpoint_path;
        rlhfConfig.clip_range = hyperparamsNode?.data.beta || 0.2; // DPO uses clip_range as beta
      } else if (algorithm === 'grpo' && grpoNode) {
        rlhfConfig.reward_model_name = grpoNode.data.model_name || 'OpenAssistant/reward-model-deberta-v3-large-v2';
        rlhfConfig.group_size = hyperparamsNode?.data.group_size || 4;
        rlhfConfig.grpo_temperature = hyperparamsNode?.data.grpo_temperature || 1.0;
        rlhfConfig.ppo_epochs = hyperparamsNode?.data.ppo_epochs || 4;
      }

      // Clear previous metrics and logs
      clearMetrics();

      // Reset training state
      updateTrainingState({
        isTraining: true,
        isPaused: false,
        progress: 0,
        currentStep: 0,
        maxSteps: rlhfConfig.max_steps,
        currentLoss: null,
        currentPPL: null,
        currentLR: null,
      });

      // Start RLHF via API
      await startRLHF(rlhfConfig);

      // Navigate to monitor
      onStartTraining();
    } catch (error: any) {
      console.error('Failed to start RLHF:', error);
      alert(`Failed to start RLHF: ${error.message}`);
      updateTrainingState({
        isTraining: false,
        isPaused: false,
      });
    }
  };

  // Detect algorithm for display
  const ppoNode = nodes.find(n => n.type === 'ppo_reward');
  const dpoNode = nodes.find(n => n.type === 'dpo_reference');
  const grpoNode = nodes.find(n => n.type === 'grpo_reward');
  const detectedAlgorithm = ppoNode ? 'PPO' : dpoNode ? 'DPO' : grpoNode ? 'GRPO' : 'None';

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
              case 'ppo_reward':
                return '#ef4444'; // red
              case 'dpo_reference':
                return '#3b82f6'; // blue
              case 'grpo_reward':
                return '#22c55e'; // green
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
              case 'rlhf_hyperparams':
                return '#f59e0b'; // amber
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
        <ValidationPanel nodes={nodes} edges={edges} mode="rlhf" />
      </div>

      {/* Configuration Panel */}
      <ConfigPanel
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onUpdate={onConfigUpdate}
      />

      {/* Start RLHF Button */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
      >
        <button
          onClick={handleStartRLHF}
          className={`px-8 py-4 rounded-xl font-bold text-lg flex items-center gap-3 shadow-2xl transition-all hover:scale-105 border-2 ${
            detectedAlgorithm === 'PPO'
              ? 'bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-700 hover:to-rose-700 border-red-400'
              : detectedAlgorithm === 'DPO'
              ? 'bg-gradient-to-r from-blue-600 to-sky-600 hover:from-blue-700 hover:to-sky-700 border-blue-400'
              : detectedAlgorithm === 'GRPO'
              ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 border-green-400'
              : 'bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-700 hover:to-slate-800 border-slate-400'
          } text-white`}
        >
          <Play className="w-6 h-6" />
          Start {detectedAlgorithm} Training
        </button>
      </motion.div>
    </div>
  );
};

const RLHFCanvasWrapper = (props: RLHFCanvasProps) => (
  <ReactFlowProvider>
    <RLHFCanvas {...props} />
  </ReactFlowProvider>
);

export default RLHFCanvasWrapper;
