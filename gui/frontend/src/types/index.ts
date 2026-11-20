// ─── Model Config ────────────────────────────────────────────────────────────

export type Architecture = 'transformer'
export type NormType = 'rmsnorm' | 'layernorm'
export type AttentionType = 'mha' | 'mqa' | 'gqa' | 'mla'
export type PositionalEncoding = 'rope' | 'sinusoidal' | 'alibi' | 'yarn'
export type Activation = 'relu' | 'relu2' | 'gelu' | 'silu' | 'swiglu' | 'geglu' | 'reglu'

export interface TransformerConfig {
  assembly_template?: string | null
  assembly?: AssemblyGraph | null
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
  use_attention_gate: boolean
  nope_every: number | ''
  rope_fraction: number
  hybrid_pattern: string
  use_sparse_attention: boolean
  sparse_top_k: number
  use_moe: boolean
  num_experts: number
  num_experts_per_token: number
  load_balancing_loss_weight: number
  router_z_loss_weight: number
  moe_router_type: 'softmax' | 'sigmoid'
  moe_aux_free_bias: boolean
  moe_shared_expert: boolean
  // MLA-specific
  d_latent: number | ''
  d_rope_latent: number | ''
  // Advanced
  tie_word_embeddings: boolean
}

export type ModelConfig = TransformerConfig

export type BrickKind = 'tokens' | 'embedding' | 'rmsnorm' | 'layernorm' | 'position' | 'attention' | 'swiglu' | 'add' | 'head' | 'repeat' | 'input_hidden' | 'output_hidden' | 'modern_attention' | 'gated_deltanet' | 'moe' | 'gemma4_norm' | 'qwen_norm' | 'ple_embedding' | 'ple_apply' | 'residual_expand' | 'gr_read' | 'gr_write' | 'gr_collapse' | 'ngram' | 'scale' | 'input_tokens' | 'input_kv' | 'input_ple' | 'output_kv'
export interface AssemblyNode {
  label?: string
  id: string
  kind: BrickKind
  params?: Record<string, number | string | boolean>
  position?: { x: number; y: number }
  body?: AssemblyGraph
}
export interface AssemblyGraph {
  version: 1 | 2
  nodes: AssemblyNode[]
  edges: { id: string; source: string; target: string; targetHandle?: string; sourceHandle?: string }[]
}

// ─── Training Config ──────────────────────────────────────────────────────────

export type Optimizer = 'adamw' | 'muon' | 'lion' | 'sophia'
export type Scheduler = 'cosine' | 'wsd' | 'linear' | 'polynomial' | 'constant' | 'adaptive'
export type LossFn = 'cross_entropy' | 'maxis'

export interface DatasetEntry {
  revision?: string
  text_field?: string
  skip?: number
  take?: number
  data_files?: string
  name: string
  split: string
  subset?: string // HF config name, e.g. "fra_Latn" for fineweb-2
  weight?: number // sampling weight when interleaving several datasets
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
  wsd_decay_fraction: number
  z_loss_weight: number
  use_compile: boolean
  use_doc_masks: boolean
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
  validation_datasets?: DatasetEntry[]
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

export type RLHFAlgorithm = 'ppo' | 'dpo' | 'grpo'

export interface RLHFConfig extends SFTConfig {
  algorithm: RLHFAlgorithm
  // PPO / GRPO — reward model scores the generated responses
  reward_model_name: string
  // DPO — frozen reference model ('' = reuse policy_checkpoint)
  reference_checkpoint: string
  mini_batch_size: number
  ppo_epochs: number
  clip_range: number // also DPO beta
  gamma: number
  gae_lambda: number
  vf_coef: number
  // GRPO
  group_size: number
  grpo_temperature: number
  grpo_clip_higher: number | '' // '' = disabled (legacy GRPO loss)
  grpo_dynamic_sampling: boolean
  grpo_importance_level: 'token' | 'sequence'
  // Generation (PPO/GRPO rollouts)
  max_new_tokens: number
  temperature: number
  top_k: number
  top_p: number
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
  trainingConfig: TrainingConfig | SFTConfig | RLHFConfig
  currentStep: number
  lastRunId?: string
}

// ─── API / Training Status ───────────────────────────────────────────────────

// Mirrors the backend training_state dict (GET /api/training/status)
export interface TrainingStatus {
  run_id?: string | null
  project_id?: string | null
  output_dir?: string | null
  created_at?: string
  checkpoint_path?: string | null
  metric_definition?: string | null
  current_language_loss?: number | null
  model_config?: Record<string, unknown> | null
  training_config?: Record<string, unknown> | null
  is_training: boolean
  current_step: number
  max_steps: number
  current_loss: number | null
  current_ppl: number | null
  current_lr: number | null
  status: string
  error: string | null
  training_type: string | null
}

export interface MetricPoint {
  step: number
  loss?: number
  language_loss?: number
  auxiliary_loss?: number
  eval_language_loss?: number
  eval_auxiliary_loss?: number
  eval_objective_loss?: number
  perplexity?: number
  eval_loss?: number
  eval_perplexity?: number
  lr?: number
  tokens_per_second?: number
}

export interface GenerateRequest {
  checkpoint_path: string
  prompt: string
  max_tokens: number
  temperature: number
  top_p: number
}

export interface GenerateResponse {
  generated_text: string
  tokens_generated: number
  tokens_per_second: number
  prompt_tokens: number
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
    use_attention_gate: false,
    nope_every: '',
    rope_fraction: 0.5,
    hybrid_pattern: '',
    use_sparse_attention: false,
    sparse_top_k: 512,
    use_moe: false,
    num_experts: 8,
    num_experts_per_token: 2,
    load_balancing_loss_weight: 0.01,
    router_z_loss_weight: 0.001,
    moe_router_type: 'softmax',
    moe_aux_free_bias: false,
    moe_shared_expert: false,
    d_latent: '',
    d_rope_latent: '',
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
    wsd_decay_fraction: 0.15,
    z_loss_weight: 0,
    use_compile: false,
    use_doc_masks: false,
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

export function defaultRLHFConfig(): RLHFConfig {
  return {
    ...defaultSFTConfig(),
    algorithm: 'grpo',
    learning_rate: 1e-6,
    weight_decay: 0.0,
    batch_size: 8,
    mini_batch_size: 4,
    gradient_accumulation_steps: 1,
    max_steps: 1000,
    reward_model_name: 'OpenAssistant/reward-model-deberta-v3-large-v2',
    reference_checkpoint: '',
    ppo_epochs: 4,
    clip_range: 0.2,
    gamma: 1.0,
    gae_lambda: 0.95,
    vf_coef: 0.1,
    group_size: 4,
    grpo_temperature: 1.0,
    grpo_clip_higher: '',
    grpo_dynamic_sampling: false,
    grpo_importance_level: 'token',
    max_new_tokens: 128,
    temperature: 1.0,
    top_k: 0,
    top_p: 1.0,
    datasets: [{ name: 'Anthropic/hh-rlhf', split: 'train' }],
  }
}
