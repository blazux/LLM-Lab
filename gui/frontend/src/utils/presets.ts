import { Node, Edge } from 'reactflow';

export interface ModelPreset {
  id: string;
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
}

export interface TrainingPreset {
  id: string;
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
}

export interface RLHFPreset {
  id: string;
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
}

// Model Architecture Presets
export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'tiny',
    name: 'Tiny Transformer (Testing)',
    description: 'Small 6-layer model for quick testing (~15M params)',
    nodes: [
      {
        id: 'tokenizer-1',
        type: 'tokenizer',
        position: { x: 250, y: 600 },
        data: { label: 'Tokenizer', tokenizer_name: 'gpt2' },
      },
      {
        id: 'embedding-1',
        type: 'embedding',
        position: { x: 250, y: 500 },
        data: { label: 'Embedding', d_model: 256, vocab_size: 50257 },
      },
      {
        id: 'rope-1',
        type: 'rope',
        position: { x: 250, y: 400 },
        data: { label: 'RoPE', max_seq_len: 512 },
      },
      {
        id: 'gqa-1',
        type: 'gqa',
        position: { x: 250, y: 300 },
        data: { label: 'GQA', n_heads: 8, n_kv_heads: 2 },
      },
      {
        id: 'rmsnorm-1',
        type: 'rmsnorm',
        position: { x: 250, y: 200 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'swiglu-1',
        type: 'swiglu',
        position: { x: 250, y: 100 },
        data: { label: 'SwiGLU', d_ff: 1024 },
      },
      {
        id: 'rmsnorm-2',
        type: 'rmsnorm',
        position: { x: 250, y: 0 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'lmhead-1',
        type: 'lmhead',
        position: { x: 250, y: -100 },
        data: { label: 'LM Head', tie_weights: true, vocab_size: 50257 },
      },
    ],
    edges: [
      { id: 'e1', source: 'tokenizer-1', target: 'embedding-1' },
      { id: 'e2', source: 'embedding-1', target: 'rope-1' },
      { id: 'e3', source: 'rope-1', target: 'gqa-1' },
      { id: 'e4', source: 'gqa-1', target: 'rmsnorm-1' },
      { id: 'e5', source: 'rmsnorm-1', target: 'swiglu-1' },
      { id: 'e6', source: 'swiglu-1', target: 'rmsnorm-2' },
      { id: 'e7', source: 'rmsnorm-2', target: 'lmhead-1' },
      // Loop edge for 6 layers (from second norm back to attention)
      {
        id: 'loop-1',
        source: 'rmsnorm-2',
        target: 'gqa-1',
        type: 'loop',
        data: { isLoop: true, repeatCount: 6 }
      },
    ],
  },
  {
    id: 'small',
    name: 'Small Transformer (GPT-2 Small)',
    description: '12-layer, 768-dim model (~125M params)',
    nodes: [
      {
        id: 'tokenizer-1',
        type: 'tokenizer',
        position: { x: 250, y: 600 },
        data: { label: 'Tokenizer', tokenizer_name: 'gpt2' },
      },
      {
        id: 'embedding-1',
        type: 'embedding',
        position: { x: 250, y: 500 },
        data: { label: 'Embedding', d_model: 768, vocab_size: 50257 },
      },
      {
        id: 'rope-1',
        type: 'rope',
        position: { x: 250, y: 400 },
        data: { label: 'RoPE', max_seq_len: 1024 },
      },
      {
        id: 'gqa-1',
        type: 'gqa',
        position: { x: 250, y: 300 },
        data: { label: 'GQA', n_heads: 12, n_kv_heads: 4 },
      },
      {
        id: 'rmsnorm-1',
        type: 'rmsnorm',
        position: { x: 250, y: 200 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'swiglu-1',
        type: 'swiglu',
        position: { x: 250, y: 100 },
        data: { label: 'SwiGLU', d_ff: 3072 },
      },
      {
        id: 'rmsnorm-2',
        type: 'rmsnorm',
        position: { x: 250, y: 0 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'lmhead-1',
        type: 'lmhead',
        position: { x: 250, y: -100 },
        data: { label: 'LM Head', tie_weights: true, vocab_size: 50257 },
      },
    ],
    edges: [
      { id: 'e1', source: 'tokenizer-1', target: 'embedding-1' },
      { id: 'e2', source: 'embedding-1', target: 'rope-1' },
      { id: 'e3', source: 'rope-1', target: 'gqa-1' },
      { id: 'e4', source: 'gqa-1', target: 'rmsnorm-1' },
      { id: 'e5', source: 'rmsnorm-1', target: 'swiglu-1' },
      { id: 'e6', source: 'swiglu-1', target: 'rmsnorm-2' },
      { id: 'e7', source: 'rmsnorm-2', target: 'lmhead-1' },
      // Loop edge for 12 layers (from second norm back to attention)
      {
        id: 'loop-1',
        source: 'rmsnorm-2',
        target: 'gqa-1',
        type: 'loop',
        data: { isLoop: true, repeatCount: 12 }
      },
    ],
  },
  {
    id: 'qwen',
    name: 'Qwen-style (0.5B)',
    description: '24-layer Qwen architecture (~0.5B params)',
    nodes: [
      {
        id: 'tokenizer-1',
        type: 'tokenizer',
        position: { x: 250, y: 600 },
        data: { label: 'Tokenizer', tokenizer_name: 'Qwen/Qwen2.5-0.5B' },
      },
      {
        id: 'embedding-1',
        type: 'embedding',
        position: { x: 250, y: 500 },
        data: { label: 'Embedding', d_model: 896, vocab_size: 151936 },
      },
      {
        id: 'rope-1',
        type: 'rope',
        position: { x: 250, y: 400 },
        data: { label: 'RoPE', max_seq_len: 1024 },
      },
      {
        id: 'gqa-1',
        type: 'gqa',
        position: { x: 250, y: 300 },
        data: { label: 'GQA', n_heads: 14, n_kv_heads: 2 },
      },
      {
        id: 'rmsnorm-1',
        type: 'rmsnorm',
        position: { x: 250, y: 200 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'swiglu-1',
        type: 'swiglu',
        position: { x: 250, y: 100 },
        data: { label: 'SwiGLU', d_ff: 4864 },
      },
      {
        id: 'rmsnorm-2',
        type: 'rmsnorm',
        position: { x: 250, y: 0 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'lmhead-1',
        type: 'lmhead',
        position: { x: 250, y: -100 },
        data: { label: 'LM Head', tie_weights: true, vocab_size: 151936 },
      },
    ],
    edges: [
      { id: 'e1', source: 'tokenizer-1', target: 'embedding-1' },
      { id: 'e2', source: 'embedding-1', target: 'rope-1' },
      { id: 'e3', source: 'rope-1', target: 'gqa-1' },
      { id: 'e4', source: 'gqa-1', target: 'rmsnorm-1' },
      { id: 'e5', source: 'rmsnorm-1', target: 'swiglu-1' },
      { id: 'e6', source: 'swiglu-1', target: 'rmsnorm-2' },
      { id: 'e7', source: 'rmsnorm-2', target: 'lmhead-1' },
      // Loop edge for 24 layers (from second norm back to attention)
      {
        id: 'loop-1',
        source: 'rmsnorm-2',
        target: 'gqa-1',
        type: 'loop',
        data: { isLoop: true, repeatCount: 24 }
      },
    ],
  },
];

// SFT Configuration Presets
export interface SFTPreset {
  id: string;
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
}

export const SFT_PRESETS: SFTPreset[] = [
  {
    id: 'sft-basic',
    name: 'Basic SFT',
    description: 'Simple supervised fine-tuning setup',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: {
          label: 'Base Model',
          checkpoint_path: 'checkpoints/best_model.pt',
        },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'HuggingFaceFW/fineweb-edu',
          split: 'train',
          weight: 1.0,
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: {
          label: 'AdamW',
          lr: 0.00001,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
        },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: {
          label: 'Cosine Scheduler',
          warmup_steps: 100,
        },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 8,
          gradient_accumulation_steps: 4,
          max_steps: 1000,
          warmup_steps: 100,
          lr: 0.0001,
          grad_clip: 1.0,
          eval_every: 100,
          eval_steps: 50,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'dataset-1', target: 'basemodel-1' },
      { id: 'e2', source: 'basemodel-1', target: 'adamw-1' },
      { id: 'e3', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e4', source: 'cosine-1', target: 'hyperparams-1' },
    ],
  },
  {
    id: 'sft-lora',
    name: 'SFT with LoRA',
    description: 'Memory-efficient fine-tuning with LoRA',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: {
          label: 'Base Model',
          checkpoint_path: 'checkpoints/best_model.pt',
        },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'HuggingFaceFW/fineweb-edu',
          split: 'train',
          weight: 1.0,
        },
      },
      {
        id: 'lora-1',
        type: 'lora',
        position: { x: 400, y: 300 },
        data: {
          label: 'LoRA',
          preset: 'attention_only',
          lora_r: 8,
          lora_alpha: 16,
          lora_dropout: 0.05,
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 200 },
        data: {
          label: 'AdamW',
          lr: 0.0001,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
        },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 100 },
        data: {
          label: 'Cosine Scheduler',
          warmup_steps: 200,
        },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 0 },
        data: {
          label: 'Hyperparameters',
          batch_size: 16,
          gradient_accumulation_steps: 2,
          max_steps: 2000,
          warmup_steps: 200,
          lr: 0.0003,
          grad_clip: 1.0,
          eval_every: 200,
          eval_steps: 50,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'dataset-1', target: 'basemodel-1' },
      { id: 'e2', source: 'basemodel-1', target: 'lora-1' },
      { id: 'e3', source: 'lora-1', target: 'adamw-1' },
      { id: 'e4', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e5', source: 'cosine-1', target: 'hyperparams-1' },
    ],
  },
];

// Training Configuration Presets
export const TRAINING_PRESETS: TrainingPreset[] = [
  {
    id: 'quick-test',
    name: 'Quick Test',
    description: 'Fast setup for testing (100 steps)',
    nodes: [
      {
        id: 'model-1',
        type: 'model',
        position: { x: 400, y: 400 },
        data: { label: 'Model', config_source: 'current', resume_training: false },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'HuggingFaceFW/fineweb-edu',
          split: 'train'
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW' },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler' },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 4,
          gradient_accumulation_steps: 1,
          max_steps: 100,
          warmup_steps: 10,
          lr: 0.001,
          grad_clip: 1.0,
          eval_every: 50,
          eval_steps: 10,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'dataset-1', target: 'model-1' },
      { id: 'e2', source: 'model-1', target: 'adamw-1' },
      { id: 'e3', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e4', source: 'cosine-1', target: 'hyperparams-1' },
    ],
  },
  {
    id: 'standard',
    name: 'Standard Training',
    description: 'Typical pretraining setup (10k steps)',
    nodes: [
      {
        id: 'model-1',
        type: 'model',
        position: { x: 400, y: 400 },
        data: { label: 'Model', config_source: 'current', resume_training: false },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'HuggingFaceFW/fineweb-edu',
          split: 'train'
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW' },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler' },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 8,
          gradient_accumulation_steps: 8,
          max_steps: 10000,
          warmup_steps: 1000,
          lr: 0.0003,
          weight_decay: 0.1,
          grad_clip: 1.0,
          eval_every: 500,
          eval_steps: 100,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'dataset-1', target: 'model-1' },
      { id: 'e2', source: 'model-1', target: 'adamw-1' },
      { id: 'e3', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e4', source: 'cosine-1', target: 'hyperparams-1' },
    ],
  },
  {
    id: 'muon-optimized',
    name: 'Muon Optimizer',
    description: 'Fast training with Muon optimizer',
    nodes: [
      {
        id: 'model-1',
        type: 'model',
        position: { x: 400, y: 400 },
        data: { label: 'Model', config_source: 'current', resume_training: false },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'HuggingFaceFW/fineweb-edu',
          split: 'train'
        },
      },
      {
        id: 'muon-1',
        type: 'muon',
        position: { x: 400, y: 300 },
        data: { label: 'Muon' },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler' },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 16,
          gradient_accumulation_steps: 4,
          max_steps: 10000,
          warmup_steps: 500,
          lr: 0.01,
          weight_decay: 0.0,
          grad_clip: 1.0,
          eval_every: 500,
          eval_steps: 100,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'dataset-1', target: 'model-1' },
      { id: 'e2', source: 'model-1', target: 'muon-1' },
      { id: 'e3', source: 'muon-1', target: 'cosine-1' },
      { id: 'e4', source: 'cosine-1', target: 'hyperparams-1' },
    ],
  },
];

// RLHF Configuration Presets
export const RLHF_PRESETS: RLHFPreset[] = [
  {
    id: 'ppo-basic',
    name: 'PPO Training',
    description: 'Proximal Policy Optimization with reward model',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: {
          label: 'Policy Model',
          checkpoint_path: 'checkpoints/best_model.pt',
        },
      },
      {
        id: 'ppo_reward-1',
        type: 'ppo_reward',
        position: { x: 400, y: 550 },
        data: {
          label: 'PPO Reward Model',
          model_name: 'OpenAssistant/reward-model-deberta-v3-large-v2',
        },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'Anthropic/hh-rlhf',
          split: 'train',
          weight: 1.0,
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: {
          label: 'AdamW',
          lr: 0.00001,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
        },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: {
          label: 'Cosine Scheduler',
          warmup_steps: 100,
        },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 128,
          mini_batch_size: 32,
          learning_rate: 0.000014,
          max_steps: 1000,
          max_new_tokens: 128,
          ppo_epochs: 4,
          clip_range: 0.2,
          gamma: 1.0,
          gae_lambda: 0.95,
          vf_coef: 0.1,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'ppo_reward-1', target: 'basemodel-1', targetHandle: 'bottom' },
      { id: 'e2', source: 'dataset-1', target: 'basemodel-1', targetHandle: 'left' },
      { id: 'e3', source: 'basemodel-1', target: 'adamw-1', sourceHandle: 'top' },
      { id: 'e4', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e5', source: 'cosine-1', target: 'rlhf_hyperparams-1' },
    ],
  },
  {
    id: 'dpo-basic',
    name: 'DPO Training',
    description: 'Direct Preference Optimization without reward model',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: {
          label: 'Policy Model',
          checkpoint_path: 'checkpoints/best_model.pt',
        },
      },
      {
        id: 'dpo_reference-1',
        type: 'dpo_reference',
        position: { x: 400, y: 550 },
        data: {
          label: 'DPO Reference Model',
          checkpoint_path: '',
        },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'Anthropic/hh-rlhf',
          split: 'train',
          weight: 1.0,
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: {
          label: 'AdamW',
          lr: 0.00001,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
        },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: {
          label: 'Cosine Scheduler',
          warmup_steps: 100,
        },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 128,
          mini_batch_size: 32,
          learning_rate: 0.000014,
          max_steps: 1000,
          max_new_tokens: 128,
          clip_range: 0.2,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'dpo_reference-1', target: 'basemodel-1', targetHandle: 'bottom' },
      { id: 'e2', source: 'dataset-1', target: 'basemodel-1', targetHandle: 'left' },
      { id: 'e3', source: 'basemodel-1', target: 'adamw-1', sourceHandle: 'top' },
      { id: 'e4', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e5', source: 'cosine-1', target: 'rlhf_hyperparams-1' },
    ],
  },
  {
    id: 'grpo-basic',
    name: 'GRPO Training',
    description: 'Group Relative Policy Optimization with reward model',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: {
          label: 'Policy Model',
          checkpoint_path: 'checkpoints/best_model.pt',
        },
      },
      {
        id: 'grpo_reward-1',
        type: 'grpo_reward',
        position: { x: 400, y: 550 },
        data: {
          label: 'GRPO Reward Model',
          model_name: 'OpenAssistant/reward-model-deberta-v3-large-v2',
        },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'Anthropic/hh-rlhf',
          split: 'train',
          weight: 1.0,
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: {
          label: 'AdamW',
          lr: 0.00001,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
        },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: {
          label: 'Cosine Scheduler',
          warmup_steps: 100,
        },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 128,
          mini_batch_size: 32,
          learning_rate: 0.000014,
          max_steps: 1000,
          max_new_tokens: 128,
          group_size: 4,
          grpo_temperature: 1.0,
          ppo_epochs: 4,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'grpo_reward-1', target: 'basemodel-1', targetHandle: 'bottom' },
      { id: 'e2', source: 'dataset-1', target: 'basemodel-1', targetHandle: 'left' },
      { id: 'e3', source: 'basemodel-1', target: 'adamw-1', sourceHandle: 'top' },
      { id: 'e4', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e5', source: 'cosine-1', target: 'rlhf_hyperparams-1' },
    ],
  },
  {
    id: 'ppo-lora',
    name: 'PPO with LoRA',
    description: 'Memory-efficient PPO training with LoRA',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: {
          label: 'Policy Model',
          checkpoint_path: 'checkpoints/best_model.pt',
        },
      },
      {
        id: 'ppo_reward-1',
        type: 'ppo_reward',
        position: { x: 400, y: 550 },
        data: {
          label: 'PPO Reward Model',
          model_name: 'OpenAssistant/reward-model-deberta-v3-large-v2',
        },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'Anthropic/hh-rlhf',
          split: 'train',
          weight: 1.0,
        },
      },
      {
        id: 'lora-1',
        type: 'lora',
        position: { x: 400, y: 300 },
        data: {
          label: 'LoRA',
          preset: 'attention_only',
          lora_r: 8,
          lora_alpha: 16,
          lora_dropout: 0.05,
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 200 },
        data: {
          label: 'AdamW',
          lr: 0.00001,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
        },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 100 },
        data: {
          label: 'Cosine Scheduler',
          warmup_steps: 100,
        },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 0 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 64,
          mini_batch_size: 16,
          learning_rate: 0.00003,
          max_steps: 2000,
          max_new_tokens: 128,
          ppo_epochs: 4,
          clip_range: 0.2,
          gamma: 1.0,
          gae_lambda: 0.95,
          vf_coef: 0.1,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'ppo_reward-1', target: 'basemodel-1', targetHandle: 'bottom' },
      { id: 'e2', source: 'dataset-1', target: 'basemodel-1', targetHandle: 'left' },
      { id: 'e3', source: 'basemodel-1', target: 'lora-1', sourceHandle: 'top' },
      { id: 'e4', source: 'lora-1', target: 'adamw-1' },
      { id: 'e5', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e6', source: 'cosine-1', target: 'rlhf_hyperparams-1' },
    ],
  },
  {
    id: 'dpo-lora',
    name: 'DPO with LoRA',
    description: 'Memory-efficient DPO training with LoRA',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: {
          label: 'Policy Model',
          checkpoint_path: 'checkpoints/best_model.pt',
        },
      },
      {
        id: 'dpo_reference-1',
        type: 'dpo_reference',
        position: { x: 400, y: 550 },
        data: {
          label: 'DPO Reference Model',
          checkpoint_path: '',
        },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: {
          label: 'Dataset',
          dataset_name: 'Anthropic/hh-rlhf',
          split: 'train',
          weight: 1.0,
        },
      },
      {
        id: 'lora-1',
        type: 'lora',
        position: { x: 400, y: 300 },
        data: {
          label: 'LoRA',
          preset: 'attention_only',
          lora_r: 8,
          lora_alpha: 16,
          lora_dropout: 0.05,
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 200 },
        data: {
          label: 'AdamW',
          lr: 0.00001,
          beta1: 0.9,
          beta2: 0.999,
          eps: 1e-8,
        },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 100 },
        data: {
          label: 'Cosine Scheduler',
          warmup_steps: 100,
        },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 0 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 64,
          mini_batch_size: 16,
          learning_rate: 0.00003,
          max_steps: 2000,
          max_new_tokens: 128,
          clip_range: 0.2,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'dpo_reference-1', target: 'basemodel-1', targetHandle: 'bottom' },
      { id: 'e2', source: 'dataset-1', target: 'basemodel-1', targetHandle: 'left' },
      { id: 'e3', source: 'basemodel-1', target: 'lora-1', sourceHandle: 'top' },
      { id: 'e4', source: 'lora-1', target: 'adamw-1' },
      { id: 'e5', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e6', source: 'cosine-1', target: 'rlhf_hyperparams-1' },
    ],
  },
];
