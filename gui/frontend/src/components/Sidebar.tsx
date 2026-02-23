import { motion, AnimatePresence } from 'framer-motion';
import {
  Blocks, Activity, X, Type, Gem, Target, MapPin, Ruler, Sparkles, GraduationCap,
  Eye, Glasses, Search, RefreshCw, Scale, Zap, Flame, Triangle, Star, Router,
  Waves, Timer, DoorOpen, Brain, Database, Rocket, Crown,
  TrendingDown, LineChart, BarChart3, Minus, Settings, CircleDot,
  Trophy, Crosshair, Users, Layers
} from 'lucide-react';
import { Node } from 'reactflow';
import { useState } from 'react';
import { MODEL_PRESETS, TRAINING_PRESETS, SFT_PRESETS, RLHF_PRESETS } from '../utils/presets';

// Icon mapping for blocks
const BLOCK_ICONS: Record<string, React.ElementType> = {
  // Core
  tokenizer: Type,
  embedding: Gem,
  lmhead: Target,
  // Positional
  rope: MapPin,
  alibi: Ruler,
  yarn: Sparkles,
  sinusoidal: Waves,
  learned: GraduationCap,
  // Attention
  mha: Eye,
  gqa: Glasses,
  mqa: Glasses,
  mla: Search,
  // Normalization
  rmsnorm: RefreshCw,
  layernorm: Scale,
  // FFN
  swiglu: Zap,
  geglu: Flame,
  reglu: Triangle,
  gelu: Star,
  relu: Triangle,
  moe_router: Router,
  // SSM
  ssmcore: Waves,
  temporalconv: Timer,
  gating: DoorOpen,
  headprojection: Brain,
  // Training
  dataset: Database,
  adamw: Zap,
  muon: Rocket,
  lion: Crown,
  sophia: GraduationCap,
  cosine: TrendingDown,
  linear: LineChart,
  polynomial: BarChart3,
  constant: Minus,
  adaptive: Activity,
  hyperparams: Settings,
  // SFT/RLHF
  basemodel: Layers,
  lora: CircleDot,
  rlhf_hyperparams: Settings,
  ppo_reward: Trophy,
  dpo_reference: Crosshair,
  grpo_reward: Users,
};

// Category colors for left-edge indicators
const CATEGORY_COLORS: Record<string, string> = {
  'Core Components': 'bg-cat-core',
  'Positional Encoding': 'bg-cat-position',
  'Attention Mechanisms': 'bg-cat-attention',
  'Normalization': 'bg-cat-norm',
  'Feed Forward': 'bg-cat-ffn',
  'State Space Model (Mamba2)': 'bg-cat-ssm',
  'Datasets': 'bg-cat-data',
  'Optimizers': 'bg-cat-optim',
  'Schedulers': 'bg-cat-sched',
  'Configuration': 'bg-cat-config',
  'Base Model': 'bg-cat-core',
  'Algorithm (choose one)': 'bg-cat-algo',
};

const MODEL_BLOCKS = [
  {
    category: 'Core Components',
    blocks: [
      { id: 'tokenizer', label: 'Tokenizer', description: 'Text to tokens' },
      { id: 'embedding', label: 'Embedding', description: 'Token to vector' },
      { id: 'lmhead', label: 'LM Head', description: 'Output projection' },
    ]
  },
  {
    category: 'Positional Encoding',
    blocks: [
      { id: 'rope', label: 'RoPE', description: 'Rotary encoding' },
      { id: 'alibi', label: 'ALiBi', description: 'Linear biases' },
      { id: 'yarn', label: 'YARN', description: 'RoPE extension' },
      { id: 'sinusoidal', label: 'Sinusoidal', description: 'Classic encoding' },
      { id: 'learned', label: 'Learned', description: 'Trainable positions' },
    ]
  },
  {
    category: 'Attention Mechanisms',
    blocks: [
      { id: 'mha', label: 'MHA', description: 'Multi-head attention' },
      { id: 'gqa', label: 'GQA', description: 'Grouped-query attention' },
      { id: 'mqa', label: 'MQA', description: 'Multi-query attention' },
      { id: 'mla', label: 'MLA', description: 'Multi-head latent' },
    ]
  },
  {
    category: 'Normalization',
    blocks: [
      { id: 'rmsnorm', label: 'RMSNorm', description: 'Fast normalization' },
      { id: 'layernorm', label: 'LayerNorm', description: 'Classic normalization' },
    ]
  },
  {
    category: 'Feed Forward',
    blocks: [
      { id: 'swiglu', label: 'SwiGLU', description: 'Modern activation' },
      { id: 'geglu', label: 'GeGLU', description: 'GELU-gated activation' },
      { id: 'reglu', label: 'ReGLU', description: 'ReLU-gated activation' },
      { id: 'gelu', label: 'GELU', description: 'Gaussian activation' },
      { id: 'relu', label: 'ReLU', description: 'Classic activation' },
      { id: 'moe_router', label: 'MoE Router', description: 'Mixture of Experts routing' },
    ]
  },
  {
    category: 'State Space Model (Mamba2)',
    blocks: [
      { id: 'ssmcore', label: 'SSM Core', description: 'State space model' },
      { id: 'temporalconv', label: 'Temporal Conv', description: 'Causal convolution' },
      { id: 'gating', label: 'Gating', description: 'Channel expansion' },
      { id: 'headprojection', label: 'Head Projection', description: 'Multi-head structure' },
    ]
  },
];

const TRAINING_BLOCKS = [
  {
    category: 'Datasets',
    blocks: [
      { id: 'dataset', label: 'Dataset', description: 'HuggingFace dataset' },
    ]
  },
  {
    category: 'Optimizers',
    blocks: [
      { id: 'adamw', label: 'AdamW', description: 'Adaptive moment estimation' },
      { id: 'muon', label: 'Muon', description: 'Momentum-based optimizer' },
      { id: 'lion', label: 'Lion', description: 'Evolved sign momentum' },
      { id: 'sophia', label: 'Sophia', description: 'Second-order optimizer' },
    ]
  },
  {
    category: 'Schedulers',
    blocks: [
      { id: 'cosine', label: 'Cosine', description: 'Cosine annealing' },
      { id: 'linear', label: 'Linear', description: 'Linear decay' },
      { id: 'polynomial', label: 'Polynomial', description: 'Polynomial decay' },
      { id: 'constant', label: 'Constant', description: 'No decay' },
      { id: 'adaptive', label: 'Adaptive', description: 'Auto-adjust based on loss' },
    ]
  },
  {
    category: 'Configuration',
    blocks: [
      { id: 'hyperparams', label: 'Hyperparameters', description: 'Batch size, steps, etc.' },
    ]
  },
];

const SFT_BLOCKS = [
  {
    category: 'Base Model',
    blocks: [
      { id: 'basemodel', label: 'Base Model', description: 'Pretrained checkpoint' },
    ]
  },
  {
    category: 'Datasets',
    blocks: [
      { id: 'dataset', label: 'Dataset', description: 'SFT dataset' },
    ]
  },
  {
    category: 'Optimizers',
    blocks: [
      { id: 'adamw', label: 'AdamW', description: 'Adaptive moment estimation' },
      { id: 'muon', label: 'Muon', description: 'Momentum-based optimizer' },
      { id: 'lion', label: 'Lion', description: 'Evolved sign momentum' },
      { id: 'sophia', label: 'Sophia', description: 'Second-order optimizer' },
    ]
  },
  {
    category: 'Schedulers',
    blocks: [
      { id: 'cosine', label: 'Cosine', description: 'Cosine annealing' },
      { id: 'linear', label: 'Linear', description: 'Linear decay' },
      { id: 'polynomial', label: 'Polynomial', description: 'Polynomial decay' },
      { id: 'constant', label: 'Constant', description: 'Fixed learning rate' },
      { id: 'adaptive', label: 'Adaptive', description: 'Auto-adjust based on loss' },
    ]
  },
  {
    category: 'Configuration',
    blocks: [
      { id: 'hyperparams', label: 'Hyperparameters', description: 'Batch size, steps, etc.' },
      { id: 'lora', label: 'LoRA', description: 'Parameter-efficient fine-tuning' },
    ]
  },
];

const RLHF_BLOCKS = [
  {
    category: 'Base Model',
    blocks: [
      { id: 'basemodel', label: 'Policy Model', description: 'Pretrained checkpoint' },
    ]
  },
  {
    category: 'Algorithm (choose one)',
    blocks: [
      { id: 'ppo_reward', label: 'PPO Reward Model', description: 'Proximal Policy Optimization' },
      { id: 'dpo_reference', label: 'DPO Reference Model', description: 'Direct Preference Optimization' },
      { id: 'grpo_reward', label: 'GRPO Reward Model', description: 'Group Relative Policy Optimization' },
    ]
  },
  {
    category: 'Datasets',
    blocks: [
      { id: 'dataset', label: 'Dataset', description: 'RLHF dataset' },
    ]
  },
  {
    category: 'Optimizers',
    blocks: [
      { id: 'adamw', label: 'AdamW', description: 'Adaptive moment estimation' },
      { id: 'muon', label: 'Muon', description: 'Momentum-based optimizer' },
      { id: 'lion', label: 'Lion', description: 'Evolved sign momentum' },
      { id: 'sophia', label: 'Sophia', description: 'Second-order optimizer' },
    ]
  },
  {
    category: 'Schedulers',
    blocks: [
      { id: 'cosine', label: 'Cosine', description: 'Cosine annealing' },
      { id: 'linear', label: 'Linear', description: 'Linear decay' },
      { id: 'polynomial', label: 'Polynomial', description: 'Polynomial decay' },
      { id: 'constant', label: 'Constant', description: 'Fixed learning rate' },
      { id: 'adaptive', label: 'Adaptive', description: 'Auto-adjust based on loss' },
    ]
  },
  {
    category: 'Configuration',
    blocks: [
      { id: 'rlhf_hyperparams', label: 'RLHF Hyperparameters', description: 'Batch size, steps, etc.' },
      { id: 'lora', label: 'LoRA', description: 'Parameter-efficient fine-tuning' },
    ]
  },
];

interface SidebarProps {
  nodes: Node[];
  onGenerateConfig: () => void;
  activeTab: 'model' | 'training' | 'sft' | 'rlhf' | 'monitor' | 'inference' | 'merge';
  onLoadPreset?: (nodes: Node[], edges: any[]) => void;
  onClearCanvas?: () => void;
  architectureFilter?: 'transformer' | 'mamba2';
  onArchitectureFilterChange?: (filter: 'transformer' | 'mamba2') => void;
}

const Sidebar = ({ nodes, onGenerateConfig, activeTab, onLoadPreset, onClearCanvas, architectureFilter = 'transformer', onArchitectureFilterChange }: SidebarProps) => {
  const [showPresetModal, setShowPresetModal] = useState(false);

  // Check which normalization type is currently used on the canvas
  const hasRMSNorm = nodes.some(node => node.type === 'rmsnorm');
  const hasLayerNorm = nodes.some(node => node.type === 'layernorm');

  // Determine which normalization blocks should be disabled
  const isNormBlockDisabled = (blockId: string) => {
    if (blockId === 'rmsnorm' && hasLayerNorm) return true;
    if (blockId === 'layernorm' && hasRMSNorm) return true;
    return false;
  };

  // Filter sections based on selected architecture
  const getFilteredSections = () => {
    if (architectureFilter === 'mamba2') {
      // For Mamba2: show Core, State Space Model, and Normalization only
      return MODEL_BLOCKS.filter(section =>
        section.category === 'Core Components' ||
        section.category === 'State Space Model (Mamba2)' ||
        section.category === 'Normalization'
      );
    } else {
      // For Transformer: show everything except State Space Model
      return MODEL_BLOCKS.filter(section =>
        section.category !== 'State Space Model (Mamba2)' &&
        section.category !== 'Architecture'
      );
    }
  };

  const onDragStart = (event: React.DragEvent, nodeType: string, label: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.setData('label', label);
    event.dataTransfer.effectAllowed = 'move';
  };

  const description = activeTab === 'model'
    ? 'Drag blocks to build your model'
    : activeTab === 'sft'
    ? 'Drag blocks to configure SFT'
    : activeTab === 'rlhf'
    ? 'Drag blocks to configure RLHF'
    : 'Drag blocks to configure training';

  return (
    <div className="w-72 bg-slate-850 border-r border-slate-700/50 flex flex-col" style={{ backgroundColor: '#1a1f2e' }}>
      {/* Header */}
      <div className="p-4 border-b border-slate-700/50">
        <h1 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <Blocks className="w-5 h-5 text-primary" />
          LLM Lab
        </h1>
        <p className="text-slate-500 text-xs mt-1">{description}</p>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto p-4">

      {/* Architecture selector for model tab */}
      {activeTab === 'model' && (
        <div className="mb-4 p-3 bg-slate-800/50 rounded-md border border-slate-700/50">
          <label className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-2 block">
            Architecture
          </label>
          <select
            value={architectureFilter}
            onChange={(e) => onArchitectureFilterChange?.(e.target.value as 'transformer' | 'mamba2')}
            className="w-full px-3 py-2 rounded text-sm bg-slate-800 text-slate-200 border border-slate-600 focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
          >
            <option value="transformer">Transformer (Attention)</option>
            <option value="mamba2">Mamba2 (State Space)</option>
          </select>
          <p className="text-xs text-slate-500 mt-2">
            {architectureFilter === 'transformer'
              ? 'O(N²) complexity'
              : 'O(N) complexity'
            }
          </p>
        </div>
      )}

      {activeTab === 'model' || activeTab === 'training' || activeTab === 'sft' || activeTab === 'rlhf' ? (
        <div className="space-y-6">
          {(activeTab === 'model' ? getFilteredSections() : activeTab === 'sft' ? SFT_BLOCKS : activeTab === 'rlhf' ? RLHF_BLOCKS : TRAINING_BLOCKS).map((section, sectionIndex) => (
            <div key={section.category}>
              <h2 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2 px-1">
                {section.category}
              </h2>
              <div className="space-y-2">
                {section.blocks.map((block, blockIndex) => {
                  const isDisabled = isNormBlockDisabled(block.id);
                  const IconComponent = BLOCK_ICONS[block.id] || Blocks;
                  const categoryColor = CATEGORY_COLORS[section.category] || 'bg-slate-500';
                  return (
                    <div
                      key={block.id}
                      draggable={!isDisabled}
                      onDragStart={(e) => !isDisabled && onDragStart(e, block.id, block.label)}
                      title={isDisabled ? `Remove all ${block.id === 'rmsnorm' ? 'LayerNorm' : 'RMSNorm'} nodes first` : ''}
                    >
                      <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: (sectionIndex * 0.3) + (blockIndex * 0.05) }}
                        className={`
                          flex items-center gap-3 px-3 py-2.5 rounded-md transition-all duration-150
                          border border-slate-600/50
                          ${isDisabled
                            ? 'cursor-not-allowed opacity-40 bg-slate-800'
                            : 'cursor-grab active:cursor-grabbing bg-slate-800 hover:bg-slate-700 hover:border-slate-500'
                          }
                        `}
                      >
                        {/* Category color indicator */}
                        <div className={`w-1 h-8 rounded-full ${categoryColor} ${isDisabled ? 'opacity-40' : ''}`} />

                        {/* Icon */}
                        <div className={`p-1.5 rounded ${isDisabled ? 'bg-slate-700' : 'bg-slate-700/80'}`}>
                          <IconComponent className={`w-4 h-4 ${isDisabled ? 'text-slate-500' : 'text-slate-300'}`} />
                        </div>

                        {/* Text */}
                        <div className="flex-1 min-w-0">
                          <div className={`font-medium text-sm ${isDisabled ? 'text-slate-500' : 'text-slate-200'}`}>
                            {block.label}
                          </div>
                          <div className={`text-xs truncate ${isDisabled ? 'text-slate-600' : 'text-slate-400'}`}>
                            {isDisabled ? 'Remove other norm type first' : block.description}
                          </div>
                        </div>
                      </motion.div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      ) : null}
      </div>

      {/* Quick Actions - Fixed at bottom */}
      <div className="p-4 border-t border-slate-700/50" style={{ backgroundColor: '#1a1f2e' }}>
        <h3 className="text-slate-400 font-medium text-xs uppercase tracking-wider mb-3 flex items-center gap-2">
          <Activity className="w-3.5 h-3.5" />
          Quick Actions
        </h3>
        <div className="space-y-2">
          {(activeTab === 'model' || activeTab === 'training' || activeTab === 'sft' || activeTab === 'rlhf') && (
            <button
              onClick={() => setShowPresetModal(true)}
              className="w-full px-3 py-2 bg-primary hover:bg-primary/90 text-white text-sm font-medium rounded transition-colors"
            >
              Load Preset
            </button>
          )}
          {(activeTab === 'model' || activeTab === 'training' || activeTab === 'sft' || activeTab === 'rlhf') && (
            <button
              onClick={onClearCanvas}
              className="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm rounded transition-colors border border-slate-600"
            >
              Clear Canvas
            </button>
          )}
          {activeTab === 'model' && (
            <button
              onClick={onGenerateConfig}
              className="w-full px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded transition-colors"
            >
              Generate Config
            </button>
          )}
        </div>
      </div>

      {/* Preset Modal */}
      <AnimatePresence>
        {showPresetModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50"
            onClick={() => setShowPresetModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-slate-800 rounded-lg p-6 max-w-lg w-full mx-4 border border-slate-700 shadow-xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-slate-100">
                  Load {activeTab === 'model' ? 'Model' : activeTab === 'sft' ? 'SFT' : activeTab === 'rlhf' ? 'RLHF' : 'Training'} Preset
                </h2>
                <button
                  onClick={() => setShowPresetModal(false)}
                  className="text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-2">
                {(activeTab === 'model' ? MODEL_PRESETS : activeTab === 'sft' ? SFT_PRESETS : activeTab === 'rlhf' ? RLHF_PRESETS : TRAINING_PRESETS).map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => {
                      if (onLoadPreset) {
                        onLoadPreset(preset.nodes, preset.edges);
                      }
                      setShowPresetModal(false);
                    }}
                    className="w-full p-3 bg-slate-700/50 hover:bg-slate-700 rounded-md text-left transition-colors border border-slate-600/50 hover:border-primary/50"
                  >
                    <div className="font-medium text-slate-200 text-sm">{preset.name}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{preset.description}</div>
                  </button>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Sidebar;
