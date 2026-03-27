// ─── Model Config ────────────────────────────────────────────────────────────

export type Architecture = 'transformer' | 'mamba2'
export type NormType = 'rmsnorm' | 'layernorm'
export type AttentionType = 'mha' | 'mqa' | 'gqa' | 'mla'
export type PositionalEncoding = 'rope' | 'sinusoidal' | 'alibi' | 'yarn'
export type Activation = 'relu' | 'gelu' | 'silu' | 'swiglu' | 'geglu' | 'reglu'

export interface TransformerConfig {
  architecture: 'transformer'
  tokenizer_name: string
  d_model: number
  n_layers: number
  vocab_size: number
  max_seq_len: number
  dropout: number
  norm_type: NormType
  norm_eps: number
  attention_type: AttentionType
  n_heads: number
  n_kv_heads: number
  positional_encoding: PositionalEncoding
  activation: Activation
  d_ff: number | ''
  sliding_window: number | ''
  attention_bias: boolean
  use_moe: boolean
  num_experts: number
  num_experts_per_token: number
  load_balancing_loss_weight: number
  router_z_loss_weight: number
  // MLA-specific
  d_latent: number | ''
  d_rope_latent: number | ''
  // Advanced
  tie_word_embeddings: boolean
}

export interface Mamba2Config {
  architecture: 'mamba2'
  tokenizer_name: string
  d_model: number
  n_layers: number
  vocab_size: number
  max_seq_len: number
  dropout: number
  norm_type: NormType
  norm_eps: number
  state_size: number
  expand_factor: number
  conv_kernel_size: number
  headdim: number
  ngroups: number
  chunk_size: number
  tie_word_embeddings: boolean
}

export type ModelConfig = TransformerConfig | Mamba2Config

// ─── Training Config ──────────────────────────────────────────────────────────

export type Optimizer = 'adamw' | 'muon' | 'lion' | 'sophia'
export type Scheduler = 'cosine' | 'linear' | 'polynomial' | 'constant' | 'adaptive'
export type LossFn = 'cross_entropy' | 'maxis'

export interface DatasetEntry {
  name: string
  split: string
}

export interface TrainingConfig {
  optimizer: Optimizer
  lr: number
  weight_decay: number
  grad_clip: number
  batch_size: number
  gradient_accumulation_steps: number
  max_steps: number
  warmup_steps: number
  scheduler: Scheduler
  eval_every: number
  eval_steps: number
  loss_fn: LossFn
  // MAXIS sub-params
  maxis_n_candidates: number
  maxis_low_rank_dim: number
  maxis_chunk_size: number
  maxis_aux_weight: number
  // Adaptive scheduler sub-params
  adaptive_window: number
  adaptive_patience: number
  adaptive_increase_factor: number
  adaptive_decrease_factor: number
  adaptive_min_lr: number
  adaptive_threshold: number
  // AdamW
  adamw_beta1: number
  adamw_beta2: number
  adamw_eps: number
  // Muon
  muon_momentum: number
  muon_nesterov: boolean
  // Lion
  lion_beta1: number
  lion_beta2: number
  // Sophia
  sophia_beta1: number
  sophia_beta2: number
  sophia_rho: number
  // Datasets
  datasets: DatasetEntry[]
  output_dir: string
  save_best_only: boolean
}

export interface SFTConfig extends TrainingConfig {
  policy_checkpoint: string
  use_lora: boolean
  lora_preset: 'minimal' | 'attention_only' | 'ffn_only' | 'all' | 'custom'
  lora_target_modules?: string[]
  lora_r: number
  lora_alpha: number
  lora_dropout: number
  learning_rate: number
  max_grad_norm: number
  log_every: number
  save_every: number
  dropout_override?: number
}

// ─── Project ──────────────────────────────────────────────────────────────────

export type ProjectType = 'pretrain' | 'sft' | 'rlhf'

export interface Project {
  id: string
  name: string
  type: ProjectType
  createdAt: string
  updatedAt: string
  modelConfig: ModelConfig
  trainingConfig: TrainingConfig | SFTConfig
  currentStep: number
}

// ─── API / Training Status ───────────────────────────────────────────────────

export interface TrainingStatus {
  is_training: boolean
  current_step: number
  total_steps: number
  loss: number | null
  lr: number | null
  tokens_per_second: number | null
  elapsed_time: number | null
  estimated_remaining: number | null
  checkpoint_path: string | null
}

export interface MetricPoint {
  step: number
  loss: number
  lr?: number
  tokens_per_second?: number
}

export interface GenerateRequest {
  checkpoint_path: string
  prompt: string
  max_new_tokens: number
  temperature: number
  top_p: number
}

export interface GenerateResponse {
  generated_text: string
  tokens_generated: number
  time_taken: number
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

export function defaultTransformerConfig(): TransformerConfig {
  return {
    architecture: 'transformer',
    tokenizer_name: 'Qwen/Qwen2.5-0.5B',
    d_model: 896,
    n_layers: 24,
    vocab_size: 151936,
    max_seq_len: 1024,
    dropout: 0.0,
    norm_type: 'rmsnorm',
    norm_eps: 1e-6,
    attention_type: 'mha',
    n_heads: 14,
    n_kv_heads: 2,
    positional_encoding: 'rope',
    activation: 'swiglu',
    d_ff: '',
    sliding_window: '',
    attention_bias: false,
    use_moe: false,
    num_experts: 8,
    num_experts_per_token: 2,
    load_balancing_loss_weight: 0.01,
    router_z_loss_weight: 0.001,
    d_latent: '',
    d_rope_latent: '',
    tie_word_embeddings: true,
  }
}

export function defaultMamba2Config(): Mamba2Config {
  return {
    architecture: 'mamba2',
    tokenizer_name: 'Qwen/Qwen2.5-0.5B',
    d_model: 896,
    n_layers: 24,
    vocab_size: 151936,
    max_seq_len: 1024,
    dropout: 0.0,
    norm_type: 'rmsnorm',
    norm_eps: 1e-6,
    state_size: 64,
    expand_factor: 2,
    conv_kernel_size: 4,
    headdim: 64,
    ngroups: 1,
    chunk_size: 256,
    tie_word_embeddings: true,
  }
}

export function defaultTrainingConfig(): TrainingConfig {
  return {
    optimizer: 'adamw',
    lr: 3e-4,
    weight_decay: 0.1,
    grad_clip: 1.0,
    batch_size: 1,
    gradient_accumulation_steps: 64,
    max_steps: 10000,
    warmup_steps: 1000,
    scheduler: 'cosine',
    eval_every: 500,
    eval_steps: 100,
    loss_fn: 'cross_entropy',
    maxis_n_candidates: 2048,
    maxis_low_rank_dim: 64,
    maxis_chunk_size: 32,
    maxis_aux_weight: 0.2,
    adaptive_window: 10,
    adaptive_patience: 3,
    adaptive_increase_factor: 1.05,
    adaptive_decrease_factor: 0.9,
    adaptive_min_lr: 1e-6,
    adaptive_threshold: 0.01,
    adamw_beta1: 0.9,
    adamw_beta2: 0.999,
    adamw_eps: 1e-8,
    muon_momentum: 0.95,
    muon_nesterov: true,
    lion_beta1: 0.9,
    lion_beta2: 0.99,
    sophia_beta1: 0.965,
    sophia_beta2: 0.99,
    sophia_rho: 0.04,
    datasets: [{ name: 'HuggingFaceFW/fineweb-edu', split: 'train' }],
    output_dir: '/app/data',
    save_best_only: true,
  }
}

export function defaultSFTConfig(): SFTConfig {
  return {
    ...defaultTrainingConfig(),
    policy_checkpoint: '',
    use_lora: false,
    lora_preset: 'attention_only',
    lora_r: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    learning_rate: 5e-6,
    max_grad_norm: 1.0,
    log_every: 10,
    save_every: 500,
  }
}
