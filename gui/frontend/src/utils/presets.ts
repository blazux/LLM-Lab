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
    id: 'nano-test',
    name: 'Nano (Smoke Test)',
    description: 'Tiny 4-layer model for pipeline testing (~20M params, trains in seconds)',
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
        data: { label: 'Embedding', d_model: 128, vocab_size: 151936 },
      },
      {
        id: 'rope-1',
        type: 'rope',
        position: { x: 250, y: 400 },
        data: { label: 'RoPE', max_seq_len: 256 },
      },
      {
        id: 'rmsnorm-1',
        type: 'rmsnorm',
        position: { x: 250, y: 300 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'gqa-1',
        type: 'gqa',
        position: { x: 250, y: 200 },
        data: { label: 'GQA', n_heads: 4, n_kv_heads: 2 },
      },
      {
        id: 'rmsnorm-2',
        type: 'rmsnorm',
        position: { x: 250, y: 100 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'swiglu-1',
        type: 'swiglu',
        position: { x: 250, y: 0 },
        data: { label: 'SwiGLU', d_ff: 384 },
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
      { id: 'e3', source: 'rope-1', target: 'rmsnorm-1' },
      { id: 'e4', source: 'rmsnorm-1', target: 'gqa-1' },
      { id: 'e5', source: 'gqa-1', target: 'rmsnorm-2' },
      { id: 'e6', source: 'rmsnorm-2', target: 'swiglu-1' },
      { id: 'e7', source: 'swiglu-1', target: 'lmhead-1' },
      {
        id: 'loop-1',
        source: 'swiglu-1',
        target: 'rmsnorm-1',
        type: 'loop',
        data: { isLoop: true, repeatCount: 4 }
      },
    ],
  },
  {
    id: 'dense-16gb',
    name: 'Dense 400M (16GB)',
    description: '24-layer GQA transformer, fits comfortably in 16GB VRAM (~400M params)',
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
        data: { label: 'Embedding', d_model: 1024, vocab_size: 151936 },
      },
      {
        id: 'rope-1',
        type: 'rope',
        position: { x: 250, y: 400 },
        data: { label: 'RoPE', max_seq_len: 2048 },
      },
      {
        id: 'rmsnorm-1',
        type: 'rmsnorm',
        position: { x: 250, y: 300 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'gqa-1',
        type: 'gqa',
        position: { x: 250, y: 200 },
        data: { label: 'GQA', n_heads: 16, n_kv_heads: 4 },
      },
      {
        id: 'rmsnorm-2',
        type: 'rmsnorm',
        position: { x: 250, y: 100 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'swiglu-1',
        type: 'swiglu',
        position: { x: 250, y: 0 },
        data: { label: 'SwiGLU', d_ff: 2816 },
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
      { id: 'e3', source: 'rope-1', target: 'rmsnorm-1' },
      { id: 'e4', source: 'rmsnorm-1', target: 'gqa-1' },
      { id: 'e5', source: 'gqa-1', target: 'rmsnorm-2' },
      { id: 'e6', source: 'rmsnorm-2', target: 'swiglu-1' },
      { id: 'e7', source: 'swiglu-1', target: 'lmhead-1' },
      {
        id: 'loop-1',
        source: 'swiglu-1',
        target: 'rmsnorm-1',
        type: 'loop',
        data: { isLoop: true, repeatCount: 24 }
      },
    ],
  },
  {
    id: 'moe-32gb',
    name: 'MoE 1B active (32GB)',
    description: '24-layer MoE transformer, 8 experts top-2 routing (~950M active / 2.5B total), fits in 32GB VRAM',
    nodes: [
      {
        id: 'tokenizer-1',
        type: 'tokenizer',
        position: { x: 250, y: 700 },
        data: { label: 'Tokenizer', tokenizer_name: 'Qwen/Qwen2.5-0.5B' },
      },
      {
        id: 'embedding-1',
        type: 'embedding',
        position: { x: 250, y: 600 },
        data: { label: 'Embedding', d_model: 1280, vocab_size: 151936 },
      },
      {
        id: 'rope-1',
        type: 'rope',
        position: { x: 250, y: 500 },
        data: { label: 'RoPE', max_seq_len: 2048 },
      },
      {
        id: 'rmsnorm-1',
        type: 'rmsnorm',
        position: { x: 250, y: 400 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'gqa-1',
        type: 'gqa',
        position: { x: 250, y: 300 },
        data: { label: 'GQA', n_heads: 20, n_kv_heads: 5 },
      },
      {
        id: 'rmsnorm-2',
        type: 'rmsnorm',
        position: { x: 250, y: 200 },
        data: { label: 'RMSNorm' },
      },
      {
        id: 'moe_router-1',
        type: 'moe_router',
        position: { x: 250, y: 100 },
        data: {
          label: 'MoE Router',
          num_experts: 8,
          num_experts_per_token: 2,
          load_balancing_loss_weight: 0.01,
          router_z_loss_weight: 0.001,
        },
      },
      {
        id: 'swiglu-1',
        type: 'swiglu',
        position: { x: 250, y: 0 },
        data: { label: 'SwiGLU (per expert)', d_ff: 3840 },
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
      { id: 'e3', source: 'rope-1', target: 'rmsnorm-1' },
      { id: 'e4', source: 'rmsnorm-1', target: 'gqa-1' },
      { id: 'e5', source: 'gqa-1', target: 'rmsnorm-2' },
      { id: 'e6', source: 'rmsnorm-2', target: 'moe_router-1' },
      { id: 'e7', source: 'moe_router-1', target: 'swiglu-1' },
      { id: 'e8', source: 'swiglu-1', target: 'lmhead-1' },
      {
        id: 'loop-1',
        source: 'swiglu-1',
        target: 'rmsnorm-1',
        type: 'loop',
        data: { isLoop: true, repeatCount: 24 }
      },
    ],
  },
  // Mamba2 Presets
  {
    id: 'tiny-mamba2',
    name: 'Tiny Mamba2 (Testing)',
    description: 'Small 6-layer Mamba2 for quick testing (~10M params)',
    nodes: [
      {
        id: 'tokenizer-1',
        type: 'tokenizer',
        position: { x: 250, y: 800 },
        data: { label: 'Tokenizer', tokenizer_name: 'gpt2' },
      },
      {
        id: 'embedding-1',
        type: 'embedding',
        position: { x: 250, y: 700 },
        data: { label: 'Embedding', d_model: 256, vocab_size: 50257 },
      },
      // Mamba2 Block components (showing the pre-norm architecture)
      {
        id: 'rmsnorm-pre-1',
        type: 'rmsnorm',
        position: { x: 250, y: 600 },
        data: { label: 'RMSNorm (pre)' },
      },
      {
        id: 'gating-1',
        type: 'gating',
        position: { x: 250, y: 500 },
        data: { label: 'Gating', expand_factor: 2 },
      },
      {
        id: 'temporalconv-1',
        type: 'temporalconv',
        position: { x: 250, y: 400 },
        data: { label: 'Temporal Conv', conv_kernel_size: 4 },
      },
      {
        id: 'ssmcore-1',
        type: 'ssmcore',
        position: { x: 250, y: 300 },
        data: { label: 'SSM Core', state_size: 32 },
      },
      {
        id: 'headprojection-1',
        type: 'headprojection',
        position: { x: 250, y: 200 },
        data: { label: 'Head Projection', headdim: 32, ngroups: 1 },
      },
      {
        id: 'rmsnorm-post-1',
        type: 'rmsnorm',
        position: { x: 250, y: 100 },
        data: { label: 'RMSNorm (post)' },
      },
      {
        id: 'lmhead-1',
        type: 'lmhead',
        position: { x: 250, y: 0 },
        data: { label: 'LM Head', tie_weights: true, vocab_size: 50257 },
      },
    ],
    edges: [
      { id: 'e1', source: 'tokenizer-1', target: 'embedding-1' },
      { id: 'e2', source: 'embedding-1', target: 'rmsnorm-pre-1' },
      { id: 'e3', source: 'rmsnorm-pre-1', target: 'gating-1' },
      { id: 'e4', source: 'gating-1', target: 'temporalconv-1' },
      { id: 'e5', source: 'temporalconv-1', target: 'ssmcore-1' },
      { id: 'e6', source: 'ssmcore-1', target: 'headprojection-1' },
      { id: 'e7', source: 'headprojection-1', target: 'rmsnorm-post-1' },
      { id: 'e8', source: 'rmsnorm-post-1', target: 'lmhead-1' },
      // Loop edge for 6 layers (from post-norm back to pre-norm)
      {
        id: 'loop-1',
        source: 'rmsnorm-post-1',
        target: 'rmsnorm-pre-1',
        type: 'loop',
        data: { isLoop: true, repeatCount: 6 }
      },
    ],
  },
  {
    id: 'small-mamba2',
    name: 'Small Mamba2 (130M)',
    description: '12-layer, 768-dim Mamba2 (~130M params)',
    nodes: [
      {
        id: 'tokenizer-1',
        type: 'tokenizer',
        position: { x: 250, y: 800 },
        data: { label: 'Tokenizer', tokenizer_name: 'gpt2' },
      },
      {
        id: 'embedding-1',
        type: 'embedding',
        position: { x: 250, y: 700 },
        data: { label: 'Embedding', d_model: 768, vocab_size: 50257 },
      },
      {
        id: 'rmsnorm-pre-1',
        type: 'rmsnorm',
        position: { x: 250, y: 600 },
        data: { label: 'RMSNorm (pre)' },
      },
      {
        id: 'gating-1',
        type: 'gating',
        position: { x: 250, y: 500 },
        data: { label: 'Gating', expand_factor: 2 },
      },
      {
        id: 'temporalconv-1',
        type: 'temporalconv',
        position: { x: 250, y: 400 },
        data: { label: 'Temporal Conv', conv_kernel_size: 4 },
      },
      {
        id: 'ssmcore-1',
        type: 'ssmcore',
        position: { x: 250, y: 300 },
        data: { label: 'SSM Core', state_size: 64 },
      },
      {
        id: 'headprojection-1',
        type: 'headprojection',
        position: { x: 250, y: 200 },
        data: { label: 'Head Projection', headdim: 64, ngroups: 1 },
      },
      {
        id: 'rmsnorm-post-1',
        type: 'rmsnorm',
        position: { x: 250, y: 100 },
        data: { label: 'RMSNorm (post)' },
      },
      {
        id: 'lmhead-1',
        type: 'lmhead',
        position: { x: 250, y: 0 },
        data: { label: 'LM Head', tie_weights: true, vocab_size: 50257 },
      },
    ],
    edges: [
      { id: 'e1', source: 'tokenizer-1', target: 'embedding-1' },
      { id: 'e2', source: 'embedding-1', target: 'rmsnorm-pre-1' },
      { id: 'e3', source: 'rmsnorm-pre-1', target: 'gating-1' },
      { id: 'e4', source: 'gating-1', target: 'temporalconv-1' },
      { id: 'e5', source: 'temporalconv-1', target: 'ssmcore-1' },
      { id: 'e6', source: 'ssmcore-1', target: 'headprojection-1' },
      { id: 'e7', source: 'headprojection-1', target: 'rmsnorm-post-1' },
      { id: 'e8', source: 'rmsnorm-post-1', target: 'lmhead-1' },
      // Loop edge for 12 layers
      {
        id: 'loop-1',
        source: 'rmsnorm-post-1',
        target: 'rmsnorm-pre-1',
        type: 'loop',
        data: { isLoop: true, repeatCount: 12 }
      },
    ],
  },
  {
    id: 'medium-mamba2',
    name: 'Medium Mamba2 (370M)',
    description: '24-layer, 1024-dim Mamba2 (~370M params)',
    nodes: [
      {
        id: 'tokenizer-1',
        type: 'tokenizer',
        position: { x: 250, y: 800 },
        data: { label: 'Tokenizer', tokenizer_name: 'gpt2' },
      },
      {
        id: 'embedding-1',
        type: 'embedding',
        position: { x: 250, y: 700 },
        data: { label: 'Embedding', d_model: 1024, vocab_size: 50257 },
      },
      {
        id: 'rmsnorm-pre-1',
        type: 'rmsnorm',
        position: { x: 250, y: 600 },
        data: { label: 'RMSNorm (pre)' },
      },
      {
        id: 'gating-1',
        type: 'gating',
        position: { x: 250, y: 500 },
        data: { label: 'Gating', expand_factor: 2 },
      },
      {
        id: 'temporalconv-1',
        type: 'temporalconv',
        position: { x: 250, y: 400 },
        data: { label: 'Temporal Conv', conv_kernel_size: 4 },
      },
      {
        id: 'ssmcore-1',
        type: 'ssmcore',
        position: { x: 250, y: 300 },
        data: { label: 'SSM Core', state_size: 128 },
      },
      {
        id: 'headprojection-1',
        type: 'headprojection',
        position: { x: 250, y: 200 },
        data: { label: 'Head Projection', headdim: 64, ngroups: 8 },
      },
      {
        id: 'rmsnorm-post-1',
        type: 'rmsnorm',
        position: { x: 250, y: 100 },
        data: { label: 'RMSNorm (post)' },
      },
      {
        id: 'lmhead-1',
        type: 'lmhead',
        position: { x: 250, y: 0 },
        data: { label: 'LM Head', tie_weights: true, vocab_size: 50257 },
      },
    ],
    edges: [
      { id: 'e1', source: 'tokenizer-1', target: 'embedding-1' },
      { id: 'e2', source: 'embedding-1', target: 'rmsnorm-pre-1' },
      { id: 'e3', source: 'rmsnorm-pre-1', target: 'gating-1' },
      { id: 'e4', source: 'gating-1', target: 'temporalconv-1' },
      { id: 'e5', source: 'temporalconv-1', target: 'ssmcore-1' },
      { id: 'e6', source: 'ssmcore-1', target: 'headprojection-1' },
      { id: 'e7', source: 'headprojection-1', target: 'rmsnorm-post-1' },
      { id: 'e8', source: 'rmsnorm-post-1', target: 'lmhead-1' },
      // Loop edge for 24 layers
      {
        id: 'loop-1',
        source: 'rmsnorm-post-1',
        target: 'rmsnorm-pre-1',
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
    id: 'sft-nano',
    name: 'Nano (Smoke Test)',
    description: 'Pipeline smoke test — 50 steps full finetune. Pair with the Nano model preset.',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: { label: 'Base Model', checkpoint_path: 'checkpoints/best_model.pt' },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: { label: 'Dataset', dataset_name: 'HuggingFaceFW/fineweb-edu', split: 'train', weight: 1.0 },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 0.001, beta1: 0.9, beta2: 0.95, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 5 },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 2,
          gradient_accumulation_steps: 1,
          max_steps: 50,
          warmup_steps: 5,
          lr: 0.001,
          grad_clip: 1.0,
          eval_every: 25,
          eval_steps: 5,
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
    id: 'sft-16gb',
    name: 'Full SFT (16GB)',
    description: 'Full finetune of the Dense 400M model — fits in 16GB without LoRA. Effective batch 32, 5k steps.',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: { label: 'Base Model', checkpoint_path: 'checkpoints/best_model.pt' },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: { label: 'Dataset', dataset_name: 'HuggingFaceFW/fineweb-edu', split: 'train', weight: 1.0 },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 2e-5, beta1: 0.9, beta2: 0.95, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 500 },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 4,
          gradient_accumulation_steps: 8,
          max_steps: 5000,
          warmup_steps: 500,
          lr: 2e-5,
          weight_decay: 0.01,
          grad_clip: 1.0,
          eval_every: 500,
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
    id: 'sft-32gb-moe',
    name: 'LoRA SFT (32GB MoE)',
    description: 'LoRA finetune of the MoE 2.5B model — full-rank SFT would not fit even at 32GB. r=32 covers attention + FFN. Effective batch 64, 5k steps.',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 500 },
        data: { label: 'Base Model', checkpoint_path: 'checkpoints/best_model.pt' },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 500 },
        data: { label: 'Dataset', dataset_name: 'HuggingFaceFW/fineweb-edu', split: 'train', weight: 1.0 },
      },
      {
        id: 'lora-1',
        type: 'lora',
        position: { x: 400, y: 400 },
        data: {
          label: 'LoRA',
          preset: 'full',
          lora_r: 32,
          lora_alpha: 64,
          lora_dropout: 0.05,
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 1e-4, beta1: 0.9, beta2: 0.95, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 500 },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 4,
          gradient_accumulation_steps: 16,
          max_steps: 5000,
          warmup_steps: 500,
          lr: 1e-4,
          weight_decay: 0.01,
          grad_clip: 1.0,
          eval_every: 500,
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
    id: 'nano-test',
    name: 'Nano (Smoke Test)',
    description: 'Pipeline smoke test — 50 steps, converges in seconds. Pair with the Nano model preset.',
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
          split: 'train',
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 0.001, beta1: 0.9, beta2: 0.95, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 5 },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 4,
          gradient_accumulation_steps: 1,
          max_steps: 50,
          warmup_steps: 5,
          lr: 0.001,
          weight_decay: 0.0,
          grad_clip: 1.0,
          eval_every: 25,
          eval_steps: 5,
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
    id: 'dense-16gb',
    name: 'Dense 400M (16GB)',
    description: 'Production pretraining for the 400M dense model. Effective batch 64, 20k steps. Pair with the Dense 400M model preset.',
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
          split: 'train',
        },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 0.0003, beta1: 0.9, beta2: 0.95, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 2000 },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 4,
          gradient_accumulation_steps: 16,
          max_steps: 20000,
          warmup_steps: 2000,
          lr: 0.0003,
          weight_decay: 0.1,
          grad_clip: 1.0,
          eval_every: 500,
          eval_steps: 50,
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
    id: 'moe-32gb',
    name: 'MoE 1B active (32GB)',
    description: 'Production pretraining for the MoE model. Effective batch 128, Muon optimizer, 20k steps. Pair with the MoE 32GB model preset.',
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
          split: 'train',
        },
      },
      {
        id: 'muon-1',
        type: 'muon',
        position: { x: 400, y: 300 },
        data: { label: 'Muon', lr: 0.02, momentum: 0.95, nesterov: true },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 2000 },
      },
      {
        id: 'hyperparams-1',
        type: 'hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'Hyperparameters',
          batch_size: 4,
          gradient_accumulation_steps: 32,
          max_steps: 20000,
          warmup_steps: 2000,
          lr: 0.02,
          weight_decay: 0.0,
          grad_clip: 1.0,
          eval_every: 500,
          eval_steps: 50,
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
  // ── DPO ──────────────────────────────────────────────────────────────────
  // DPO is the lightest RLHF variant: policy + frozen reference copy, no
  // value model.  The Dense 400M (policy ~800 MB bf16 × 2 for ref) comfortably
  // fits in 16 GB without LoRA.
  {
    id: 'dpo-16gb',
    name: 'DPO (16GB)',
    description: 'Direct Preference Optimization — Dense 400M, no LoRA. Policy + frozen ref fit in 16GB. Effective batch 64, 3k steps.',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 400 },
        data: { label: 'Policy Model', checkpoint_path: 'checkpoints/best_model.pt' },
      },
      {
        id: 'dpo_reference-1',
        type: 'dpo_reference',
        position: { x: 400, y: 550 },
        data: { label: 'Reference Model', checkpoint_path: '' },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 400 },
        data: { label: 'Dataset', dataset_name: 'Anthropic/hh-rlhf', split: 'train', weight: 1.0 },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 1e-5, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 300 },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 4,
          mini_batch_size: 4,
          learning_rate: 1e-5,
          max_steps: 3000,
          max_new_tokens: 256,
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
    id: 'dpo-lora',
    name: 'DPO + LoRA (32GB MoE)',
    description: 'DPO on the MoE 2.5B model with LoRA — only the adapter is trained, the frozen reference shares base weights. Effective batch 64, 3k steps.',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 500 },
        data: { label: 'Policy Model', checkpoint_path: 'checkpoints/best_model.pt' },
      },
      {
        id: 'dpo_reference-1',
        type: 'dpo_reference',
        position: { x: 400, y: 650 },
        data: { label: 'Reference Model', checkpoint_path: '' },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 500 },
        data: { label: 'Dataset', dataset_name: 'Anthropic/hh-rlhf', split: 'train', weight: 1.0 },
      },
      {
        id: 'lora-1',
        type: 'lora',
        position: { x: 400, y: 400 },
        data: { label: 'LoRA', preset: 'attention_only', lora_r: 16, lora_alpha: 32, lora_dropout: 0.05 },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 2e-5, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 300 },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 4,
          mini_batch_size: 4,
          learning_rate: 2e-5,
          max_steps: 3000,
          max_new_tokens: 256,
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
  // ── GRPO ─────────────────────────────────────────────────────────────────
  // GRPO samples a group of completions per prompt and normalises rewards
  // within the group — no separate value model, but group rollouts multiply
  // activation memory.  LoRA is recommended even at 16 GB.
  {
    id: 'grpo-16gb',
    name: 'GRPO (16GB)',
    description: 'GRPO on the Dense 400M model with LoRA — group rollouts (8×) need the extra headroom LoRA frees up. Effective batch 16, 2k steps.',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 500 },
        data: { label: 'Policy Model', checkpoint_path: 'checkpoints/best_model.pt' },
      },
      {
        id: 'grpo_reward-1',
        type: 'grpo_reward',
        position: { x: 400, y: 650 },
        data: { label: 'Reward Model', model_name: 'OpenAssistant/reward-model-deberta-v3-large-v2' },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 500 },
        data: { label: 'Dataset', dataset_name: 'Anthropic/hh-rlhf', split: 'train', weight: 1.0 },
      },
      {
        id: 'lora-1',
        type: 'lora',
        position: { x: 400, y: 400 },
        data: { label: 'LoRA', preset: 'attention_only', lora_r: 16, lora_alpha: 32, lora_dropout: 0.05 },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 5e-6, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 200 },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 2,
          mini_batch_size: 2,
          learning_rate: 5e-6,
          max_steps: 2000,
          max_new_tokens: 256,
          group_size: 8,
          grpo_temperature: 0.9,
          ppo_epochs: 1,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'grpo_reward-1', target: 'basemodel-1', targetHandle: 'bottom' },
      { id: 'e2', source: 'dataset-1', target: 'basemodel-1', targetHandle: 'left' },
      { id: 'e3', source: 'basemodel-1', target: 'lora-1', sourceHandle: 'top' },
      { id: 'e4', source: 'lora-1', target: 'adamw-1' },
      { id: 'e5', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e6', source: 'cosine-1', target: 'rlhf_hyperparams-1' },
    ],
  },
  {
    id: 'grpo-32gb-moe',
    name: 'GRPO + LoRA (32GB MoE)',
    description: 'GRPO on the MoE 2.5B model — LoRA r=32 keeps peak memory under 30 GB with group_size=8 rollouts. Effective batch 16, 2k steps.',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 500 },
        data: { label: 'Policy Model', checkpoint_path: 'checkpoints/best_model.pt' },
      },
      {
        id: 'grpo_reward-1',
        type: 'grpo_reward',
        position: { x: 400, y: 650 },
        data: { label: 'Reward Model', model_name: 'OpenAssistant/reward-model-deberta-v3-large-v2' },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 500 },
        data: { label: 'Dataset', dataset_name: 'Anthropic/hh-rlhf', split: 'train', weight: 1.0 },
      },
      {
        id: 'lora-1',
        type: 'lora',
        position: { x: 400, y: 400 },
        data: { label: 'LoRA', preset: 'attention_only', lora_r: 32, lora_alpha: 64, lora_dropout: 0.05 },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 5e-6, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 200 },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 2,
          mini_batch_size: 2,
          learning_rate: 5e-6,
          max_steps: 2000,
          max_new_tokens: 256,
          group_size: 8,
          grpo_temperature: 0.9,
          ppo_epochs: 1,
        },
      },
    ],
    edges: [
      { id: 'e1', source: 'grpo_reward-1', target: 'basemodel-1', targetHandle: 'bottom' },
      { id: 'e2', source: 'dataset-1', target: 'basemodel-1', targetHandle: 'left' },
      { id: 'e3', source: 'basemodel-1', target: 'lora-1', sourceHandle: 'top' },
      { id: 'e4', source: 'lora-1', target: 'adamw-1' },
      { id: 'e5', source: 'adamw-1', target: 'cosine-1' },
      { id: 'e6', source: 'cosine-1', target: 'rlhf_hyperparams-1' },
    ],
  },
  // ── PPO ──────────────────────────────────────────────────────────────────
  // PPO requires a value head on top of the policy (effectively 2 models in
  // memory).  LoRA is essentially mandatory to keep VRAM manageable.
  {
    id: 'ppo-lora',
    name: 'PPO + LoRA (16GB)',
    description: 'PPO on the Dense 400M model — LoRA frees enough VRAM for the value head. Standard PPO with 4 update epochs per rollout batch.',
    nodes: [
      {
        id: 'basemodel-1',
        type: 'basemodel',
        position: { x: 400, y: 500 },
        data: { label: 'Policy Model', checkpoint_path: 'checkpoints/best_model.pt' },
      },
      {
        id: 'ppo_reward-1',
        type: 'ppo_reward',
        position: { x: 400, y: 650 },
        data: { label: 'Reward Model', model_name: 'OpenAssistant/reward-model-deberta-v3-large-v2' },
      },
      {
        id: 'dataset-1',
        type: 'dataset',
        position: { x: 100, y: 500 },
        data: { label: 'Dataset', dataset_name: 'Anthropic/hh-rlhf', split: 'train', weight: 1.0 },
      },
      {
        id: 'lora-1',
        type: 'lora',
        position: { x: 400, y: 400 },
        data: { label: 'LoRA', preset: 'attention_only', lora_r: 8, lora_alpha: 16, lora_dropout: 0.05 },
      },
      {
        id: 'adamw-1',
        type: 'adamw',
        position: { x: 400, y: 300 },
        data: { label: 'AdamW', lr: 1e-5, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
      },
      {
        id: 'cosine-1',
        type: 'cosine',
        position: { x: 400, y: 200 },
        data: { label: 'Cosine Scheduler', warmup_steps: 200 },
      },
      {
        id: 'rlhf_hyperparams-1',
        type: 'rlhf_hyperparams',
        position: { x: 400, y: 100 },
        data: {
          label: 'RLHF Hyperparameters',
          batch_size: 4,
          mini_batch_size: 4,
          learning_rate: 1e-5,
          max_steps: 2000,
          max_new_tokens: 256,
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
];
