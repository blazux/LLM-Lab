import { motion, AnimatePresence } from 'framer-motion';
import { Blocks, Activity, X } from 'lucide-react';
import { Node } from 'reactflow';
import { useState } from 'react';
import { MODEL_PRESETS, TRAINING_PRESETS, SFT_PRESETS, RLHF_PRESETS } from '../utils/presets';

const MODEL_BLOCKS = [
  {
    category: 'Core Components',
    blocks: [
      {
        id: 'tokenizer',
        label: 'Tokenizer',
        emoji: '🔤',
        color: 'from-purple-500 to-purple-700',
        description: 'Text to tokens'
      },
      {
        id: 'embedding',
        label: 'Embedding',
        emoji: '💎',
        color: 'from-teal-500 to-teal-700',
        description: 'Token to vector'
      },
      {
        id: 'lmhead',
        label: 'LM Head',
        emoji: '🎯',
        color: 'from-pink-500 to-pink-700',
        description: 'Output projection'
      },
    ]
  },
  {
    category: 'Positional Encoding',
    blocks: [
      {
        id: 'rope',
        label: 'RoPE',
        emoji: '📍',
        color: 'from-blue-500 to-blue-700',
        description: 'Rotary encoding'
      },
      {
        id: 'alibi',
        label: 'ALiBi',
        emoji: '📐',
        color: 'from-sky-500 to-sky-700',
        description: 'Linear biases'
      },
      {
        id: 'yarn',
        label: 'YARN',
        emoji: '🧶',
        color: 'from-indigo-500 to-indigo-700',
        description: 'RoPE extension'
      },
      {
        id: 'sinusoidal',
        label: 'Sinusoidal',
        emoji: '〰️',
        color: 'from-violet-500 to-violet-700',
        description: 'Classic encoding'
      },
    ]
  },
  {
    category: 'Attention Mechanisms',
    blocks: [
      {
        id: 'mha',
        label: 'MHA',
        emoji: '👁️',
        color: 'from-green-500 to-green-700',
        description: 'Multi-head attention'
      },
      {
        id: 'gqa',
        label: 'GQA',
        emoji: '👀',
        color: 'from-emerald-500 to-emerald-700',
        description: 'Grouped-query attention'
      },
      {
        id: 'mqa',
        label: 'MQA',
        emoji: '👓',
        color: 'from-teal-500 to-teal-700',
        description: 'Multi-query attention'
      },
      {
        id: 'mla',
        label: 'MLA',
        emoji: '🔍',
        color: 'from-cyan-500 to-cyan-700',
        description: 'Multi-head latent'
      },
    ]
  },
  {
    category: 'Normalization',
    blocks: [
      {
        id: 'rmsnorm',
        label: 'RMSNorm',
        emoji: '🔄',
        color: 'from-yellow-500 to-yellow-700',
        description: 'Fast normalization'
      },
      {
        id: 'layernorm',
        label: 'LayerNorm',
        emoji: '⚖️',
        color: 'from-amber-500 to-amber-700',
        description: 'Classic normalization'
      },
    ]
  },
  {
    category: 'Feed Forward',
    blocks: [
      {
        id: 'swiglu',
        label: 'SwiGLU',
        emoji: '⚡',
        color: 'from-orange-500 to-orange-700',
        description: 'Modern activation'
      },
      {
        id: 'gelu',
        label: 'GELU',
        emoji: '💫',
        color: 'from-red-500 to-red-700',
        description: 'Gaussian activation'
      },
      {
        id: 'relu',
        label: 'ReLU',
        emoji: '🔺',
        color: 'from-pink-500 to-pink-700',
        description: 'Classic activation'
      },
    ]
  },
  {
    category: 'State Space Model (Mamba2)',
    blocks: [
      {
        id: 'ssmcore',
        label: 'SSM Core',
        emoji: '🌊',
        color: 'from-cyan-500 to-cyan-700',
        description: 'State space model'
      },
      {
        id: 'temporalconv',
        label: 'Temporal Conv',
        emoji: '⏱️',
        color: 'from-teal-500 to-teal-700',
        description: 'Causal convolution'
      },
      {
        id: 'gating',
        label: 'Gating',
        emoji: '🚪',
        color: 'from-sky-500 to-sky-700',
        description: 'Channel expansion'
      },
      {
        id: 'headprojection',
        label: 'Head Projection',
        emoji: '🧠',
        color: 'from-indigo-500 to-indigo-700',
        description: 'Multi-head structure'
      },
    ]
  },
];

const TRAINING_BLOCKS = [
  {
    category: 'Datasets',
    blocks: [
      {
        id: 'dataset',
        label: 'Dataset',
        emoji: '📚',
        color: 'from-blue-500 to-blue-700',
        description: 'HuggingFace dataset'
      },
    ]
  },
  {
    category: 'Optimizers',
    blocks: [
      {
        id: 'adamw',
        label: 'AdamW',
        emoji: '⚡',
        color: 'from-red-500 to-red-700',
        description: 'Adaptive moment estimation'
      },
      {
        id: 'muon',
        label: 'Muon',
        emoji: '🚀',
        color: 'from-orange-500 to-orange-700',
        description: 'Momentum-based optimizer'
      },
      {
        id: 'lion',
        label: 'Lion',
        emoji: '🦁',
        color: 'from-amber-500 to-amber-700',
        description: 'Evolved sign momentum'
      },
      {
        id: 'sophia',
        label: 'Sophia',
        emoji: '🎓',
        color: 'from-rose-500 to-rose-700',
        description: 'Second-order optimizer'
      },
    ]
  },
  {
    category: 'Schedulers',
    blocks: [
      {
        id: 'cosine',
        label: 'Cosine',
        emoji: '〰️',
        color: 'from-purple-500 to-purple-700',
        description: 'Cosine annealing'
      },
      {
        id: 'linear',
        label: 'Linear',
        emoji: '📉',
        color: 'from-violet-500 to-violet-700',
        description: 'Linear decay'
      },
      {
        id: 'polynomial',
        label: 'Polynomial',
        emoji: '📊',
        color: 'from-fuchsia-500 to-fuchsia-700',
        description: 'Polynomial decay'
      },
      {
        id: 'constant',
        label: 'Constant',
        emoji: '➖',
        color: 'from-slate-500 to-slate-700',
        description: 'No decay'
      },
    ]
  },
  {
    category: 'Configuration',
    blocks: [
      {
        id: 'hyperparams',
        label: 'Hyperparameters',
        emoji: '⚙️',
        color: 'from-green-500 to-green-700',
        description: 'Batch size, steps, etc.'
      },
    ]
  },
];

const SFT_BLOCKS = [
  {
    category: 'Base Model',
    blocks: [
      {
        id: 'basemodel',
        label: 'Base Model',
        emoji: '🏗️',
        color: 'from-blue-500 to-blue-700',
        description: 'Pretrained checkpoint'
      },
    ]
  },
  {
    category: 'Datasets',
    blocks: [
      {
        id: 'dataset',
        label: 'Dataset',
        emoji: '📚',
        color: 'from-blue-500 to-blue-700',
        description: 'SFT dataset'
      },
    ]
  },
  {
    category: 'Optimizers',
    blocks: [
      {
        id: 'adamw',
        label: 'AdamW',
        emoji: '⚡',
        color: 'from-red-500 to-red-700',
        description: 'Adaptive moment estimation'
      },
      {
        id: 'muon',
        label: 'Muon',
        emoji: '🚀',
        color: 'from-orange-500 to-orange-700',
        description: 'Momentum-based optimizer'
      },
      {
        id: 'lion',
        label: 'Lion',
        emoji: '🦁',
        color: 'from-amber-500 to-amber-700',
        description: 'Evolved sign momentum'
      },
      {
        id: 'sophia',
        label: 'Sophia',
        emoji: '🎓',
        color: 'from-rose-500 to-rose-700',
        description: 'Second-order optimizer'
      },
    ]
  },
  {
    category: 'Schedulers',
    blocks: [
      {
        id: 'cosine',
        label: 'Cosine',
        emoji: '📉',
        color: 'from-purple-500 to-purple-700',
        description: 'Cosine annealing'
      },
      {
        id: 'linear',
        label: 'Linear',
        emoji: '📊',
        color: 'from-violet-500 to-violet-700',
        description: 'Linear decay'
      },
      {
        id: 'polynomial',
        label: 'Polynomial',
        emoji: '📈',
        color: 'from-fuchsia-500 to-fuchsia-700',
        description: 'Polynomial decay'
      },
      {
        id: 'constant',
        label: 'Constant',
        emoji: '➡️',
        color: 'from-slate-500 to-slate-700',
        description: 'Fixed learning rate'
      },
    ]
  },
  {
    category: 'Configuration',
    blocks: [
      {
        id: 'hyperparams',
        label: 'Hyperparameters',
        emoji: '⚙️',
        color: 'from-green-500 to-green-700',
        description: 'Batch size, steps, etc.'
      },
      {
        id: 'lora',
        label: 'LoRA',
        emoji: '🎯',
        color: 'from-purple-500 to-purple-700',
        description: 'Parameter-efficient fine-tuning'
      },
    ]
  },
];

const RLHF_BLOCKS = [
  {
    category: 'Base Model',
    blocks: [
      {
        id: 'basemodel',
        label: 'Policy Model',
        emoji: '🏗️',
        color: 'from-blue-500 to-blue-700',
        description: 'Pretrained checkpoint'
      },
    ]
  },
  {
    category: 'Algorithm (choose one)',
    blocks: [
      {
        id: 'ppo_reward',
        label: 'PPO Reward Model',
        emoji: '🏆',
        color: 'from-red-500 to-red-700',
        description: 'Proximal Policy Optimization'
      },
      {
        id: 'dpo_reference',
        label: 'DPO Reference Model',
        emoji: '🎯',
        color: 'from-blue-500 to-blue-700',
        description: 'Direct Preference Optimization'
      },
      {
        id: 'grpo_reward',
        label: 'GRPO Reward Model',
        emoji: '👥',
        color: 'from-green-500 to-green-700',
        description: 'Group Relative Policy Optimization'
      },
    ]
  },
  {
    category: 'Datasets',
    blocks: [
      {
        id: 'dataset',
        label: 'Dataset',
        emoji: '📚',
        color: 'from-blue-500 to-blue-700',
        description: 'RLHF dataset'
      },
    ]
  },
  {
    category: 'Optimizers',
    blocks: [
      {
        id: 'adamw',
        label: 'AdamW',
        emoji: '⚡',
        color: 'from-red-500 to-red-700',
        description: 'Adaptive moment estimation'
      },
      {
        id: 'muon',
        label: 'Muon',
        emoji: '🚀',
        color: 'from-orange-500 to-orange-700',
        description: 'Momentum-based optimizer'
      },
      {
        id: 'lion',
        label: 'Lion',
        emoji: '🦁',
        color: 'from-amber-500 to-amber-700',
        description: 'Evolved sign momentum'
      },
      {
        id: 'sophia',
        label: 'Sophia',
        emoji: '🎓',
        color: 'from-rose-500 to-rose-700',
        description: 'Second-order optimizer'
      },
    ]
  },
  {
    category: 'Schedulers',
    blocks: [
      {
        id: 'cosine',
        label: 'Cosine',
        emoji: '📉',
        color: 'from-purple-500 to-purple-700',
        description: 'Cosine annealing'
      },
      {
        id: 'linear',
        label: 'Linear',
        emoji: '📊',
        color: 'from-violet-500 to-violet-700',
        description: 'Linear decay'
      },
      {
        id: 'polynomial',
        label: 'Polynomial',
        emoji: '📈',
        color: 'from-fuchsia-500 to-fuchsia-700',
        description: 'Polynomial decay'
      },
      {
        id: 'constant',
        label: 'Constant',
        emoji: '➡️',
        color: 'from-slate-500 to-slate-700',
        description: 'Fixed learning rate'
      },
    ]
  },
  {
    category: 'Configuration',
    blocks: [
      {
        id: 'rlhf_hyperparams',
        label: 'RLHF Hyperparameters',
        emoji: '⚙️',
        color: 'from-amber-500 to-amber-700',
        description: 'Batch size, steps, etc.'
      },
      {
        id: 'lora',
        label: 'LoRA',
        emoji: '🎯',
        color: 'from-purple-500 to-purple-700',
        description: 'Parameter-efficient fine-tuning'
      },
    ]
  },
];

interface SidebarProps {
  nodes: Node[];
  onGenerateConfig: () => void;
  activeTab: 'model' | 'training' | 'sft' | 'rlhf' | 'monitor' | 'inference';
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
    <div className="w-80 bg-slate-800 border-r border-slate-700 p-6 overflow-y-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
          <Blocks className="text-purple-400" />
          LLM Lab
        </h1>
        <p className="text-slate-400 text-sm">{description}</p>
      </div>

      {/* Architecture selector for model tab */}
      {activeTab === 'model' && (
        <div className="mb-6 p-4 bg-slate-900 rounded-lg border border-slate-700">
          <label className="text-white text-sm font-semibold mb-2 block">
            Architecture
          </label>
          <select
            value={architectureFilter}
            onChange={(e) => onArchitectureFilterChange?.(e.target.value as 'transformer' | 'mamba2')}
            className="w-full px-3 py-2 rounded-md text-sm bg-slate-800 text-white border border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500"
          >
            <option value="transformer">🏛️ Transformer (Attention-based)</option>
            <option value="mamba2">🌊 Mamba2 (State Space Model)</option>
          </select>
          <p className="text-xs text-slate-400 mt-2">
            {architectureFilter === 'transformer'
              ? 'O(N²) complexity - Uses attention mechanisms'
              : 'O(N) complexity - Uses state space models'
            }
          </p>
        </div>
      )}

      {activeTab === 'model' || activeTab === 'training' || activeTab === 'sft' || activeTab === 'rlhf' ? (
        <div className="space-y-6">
          {(activeTab === 'model' ? getFilteredSections() : activeTab === 'sft' ? SFT_BLOCKS : activeTab === 'rlhf' ? RLHF_BLOCKS : TRAINING_BLOCKS).map((section, sectionIndex) => (
            <div key={section.category}>
              <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
                {section.category}
              </h2>
              <div className="space-y-3">
                {section.blocks.map((block, blockIndex) => {
                  const isDisabled = isNormBlockDisabled(block.id);
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
                        transition={{ delay: (sectionIndex * 0.5) + (blockIndex * 0.1) }}
                        className={`
                          p-4 rounded-lg transition-all duration-200
                          ${isDisabled
                            ? 'cursor-not-allowed opacity-40 bg-gradient-to-br from-slate-600 to-slate-700 border-2 border-slate-500'
                            : `cursor-grab active:cursor-grabbing bg-gradient-to-br ${block.color} border-2 border-opacity-50 hover:scale-105 shadow-lg hover:shadow-xl`
                          }
                        `}
                      >
                        <div className="flex items-start gap-3">
                          <div className="text-3xl">{block.emoji}</div>
                          <div className="flex-1">
                            <div className={`font-semibold text-sm mb-1 ${isDisabled ? 'text-slate-400' : 'text-white'}`}>
                              {block.label}
                            </div>
                            <div className={`text-xs ${isDisabled ? 'text-slate-500' : 'text-white opacity-80'}`}>
                              {isDisabled ? '⚠️ Remove other norm type first' : block.description}
                            </div>
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

      <div className="mt-8 p-4 bg-slate-900 rounded-lg border border-slate-700">
        <h3 className="text-white font-semibold text-sm mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Quick Actions
        </h3>
        <div className="space-y-2">
          {(activeTab === 'model' || activeTab === 'training' || activeTab === 'sft' || activeTab === 'rlhf') && (
            <button
              onClick={() => setShowPresetModal(true)}
              className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white text-sm rounded-md transition-colors"
            >
              Load Preset
            </button>
          )}
          {(activeTab === 'model' || activeTab === 'training' || activeTab === 'sft' || activeTab === 'rlhf') && (
            <button
              onClick={onClearCanvas}
              className="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm rounded-md transition-colors"
            >
              Clear Canvas
            </button>
          )}
          {activeTab === 'model' && (
            <button
              onClick={onGenerateConfig}
              className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md transition-colors"
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
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setShowPresetModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-slate-800 rounded-lg p-6 max-w-2xl w-full mx-4 border border-slate-700"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-white">
                  Load {activeTab === 'model' ? 'Model' : activeTab === 'sft' ? 'SFT' : activeTab === 'rlhf' ? 'RLHF' : 'Training'} Preset
                </h2>
                <button
                  onClick={() => setShowPresetModal(false)}
                  className="text-slate-400 hover:text-white"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="space-y-3">
                {(activeTab === 'model' ? MODEL_PRESETS : activeTab === 'sft' ? SFT_PRESETS : activeTab === 'rlhf' ? RLHF_PRESETS : TRAINING_PRESETS).map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => {
                      if (onLoadPreset) {
                        onLoadPreset(preset.nodes, preset.edges);
                      }
                      setShowPresetModal(false);
                    }}
                    className="w-full p-4 bg-slate-700 hover:bg-slate-600 rounded-lg text-left transition-colors border border-slate-600 hover:border-purple-500"
                  >
                    <div className="font-semibold text-white mb-1">{preset.name}</div>
                    <div className="text-sm text-slate-400">{preset.description}</div>
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
