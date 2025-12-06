// Use relative URL in production (Docker), absolute URL in development
const API_BASE_URL = import.meta.env.PROD ? '/api' : 'http://localhost:8000/api';

export interface TrainingMetric {
  type: 'metrics' | 'log' | 'status' | 'eval_metrics';
  step?: number;
  loss?: number;
  lr?: number;
  perplexity?: number;
  eval_loss?: number;
  eval_perplexity?: number;
  timestamp: number;
  // For logs
  level?: 'info' | 'warning' | 'error' | 'success';
  message?: string;
  // For status
  status?: string;
  error?: string | null;
}

export interface ModelConfigData {
  model_architecture: string;
  tokenizer_name: string;
  d_model: number;
  n_layers: number;
  vocab_size: number;
  max_seq_len: number;
  positional_encoding: string;
  attention_type: string;
  activation: string;
  n_heads: number;
  n_kv_heads: number;
  d_ff: number;
  norm_type: string;
  norm_eps: number;
  dropout: number;
  tie_word_embeddings: boolean;
  model_type: string;
  // MLA-specific (optional)
  d_latent?: number;
  d_rope_latent?: number;
}

export interface TrainingConfigData {
  datasets: Array<{
    name: string;
    subset?: string;
    split: string;
    weight: number;
  }>;
  optimizer: string;
  lr: number;
  weight_decay: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  max_steps: number;
  warmup_steps: number;
  scheduler: string;
  grad_clip: number;
  eval_every: number;
  eval_steps: number;
  // Optimizer-specific params
  adamw_beta1?: number;
  adamw_beta2?: number;
  adamw_eps?: number;
  muon_momentum?: number;
  muon_nesterov?: boolean;
  lion_beta1?: number;
  lion_beta2?: number;
  sophia_beta1?: number;
  sophia_beta2?: number;
  sophia_rho?: number;
}

export interface SFTConfigData {
  policy_checkpoint: string;
  datasets: Array<{
    name: string;
    subset?: string;
    split: string;
    weight: number;
  }>;
  optimizer: string;
  lr: number;
  weight_decay: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  max_steps: number;
  warmup_steps: number;
  scheduler: string;
  max_grad_norm: number;
  log_every: number;
  save_every: number;
  eval_every: number;
  eval_steps: number;
  save_best_only: boolean;
  output_dir: string;
  // LoRA configuration
  use_lora: boolean;
  lora_preset?: string;
  lora_target_modules?: string[];
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  // Optimizer-specific params
  adamw_beta1?: number;
  adamw_beta2?: number;
  adamw_eps?: number;
  muon_momentum?: number;
  muon_nesterov?: boolean;
  lion_beta1?: number;
  lion_beta2?: number;
  sophia_beta1?: number;
  sophia_beta2?: number;
  sophia_rho?: number;
}

export async function startTraining(
  modelConfig: ModelConfigData,
  trainingConfig: TrainingConfigData,
  checkpointPath?: string,
  outputDir: string = 'checkpoints'
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/training/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_cfg: modelConfig,
      training_cfg: trainingConfig,
      checkpoint_path: checkpointPath,
      output_dir: outputDir,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to start training');
  }

  return response.json();
}

export async function startSFT(sftConfig: SFTConfigData): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/training/sft/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(sftConfig),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to start SFT');
  }

  return response.json();
}

export interface RLHFConfigData {
  algorithm: 'ppo' | 'dpo' | 'grpo';
  policy_checkpoint: string;
  datasets: Array<{
    name: string;
    subset?: string;
    split: string;
    weight: number;
  }>;
  optimizer: string;
  learning_rate: number;
  weight_decay: number;
  batch_size: number;
  mini_batch_size: number;
  gradient_accumulation_steps: number;
  max_steps: number;
  max_grad_norm: number;
  log_every: number;
  save_every: number;
  eval_every: number;
  output_dir: string;
  // Generation parameters
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  // LoRA configuration
  use_lora: boolean;
  lora_preset?: string;
  lora_target_modules?: string[];
  lora_r?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  // Optimizer-specific params
  adamw_beta1?: number;
  adamw_beta2?: number;
  adamw_eps?: number;
  muon_momentum?: number;
  muon_nesterov?: boolean;
  lion_beta1?: number;
  lion_beta2?: number;
  sophia_beta1?: number;
  sophia_beta2?: number;
  sophia_rho?: number;
  // Algorithm-specific params
  reward_model_name?: string;
  reference_checkpoint?: string;
  ppo_epochs?: number;
  clip_range?: number;
  gamma?: number;
  gae_lambda?: number;
  vf_coef?: number;
  group_size?: number;
  grpo_temperature?: number;
}

export async function startRLHF(rlhfConfig: RLHFConfigData): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/training/rlhf/start`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(rlhfConfig),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to start RLHF');
  }

  return response.json();
}

export async function stopTraining(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/training/stop`, {
    method: 'POST',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to stop training');
  }

  return response.json();
}

export async function getTrainingStatus(): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/training/status`);
  if (!response.ok) {
    throw new Error('Failed to get training status');
  }
  return response.json();
}

export function subscribeToMetrics(
  onMetric: (metric: TrainingMetric) => void,
  onError?: (error: Error) => void
): () => void {
  const eventSource = new EventSource(`${API_BASE_URL}/training/metrics/stream`);

  eventSource.onmessage = (event) => {
    try {
      const metric: TrainingMetric = JSON.parse(event.data);
      onMetric(metric);
    } catch (error) {
      console.error('Failed to parse metric:', error);
      if (onError) onError(error as Error);
    }
  };

  eventSource.addEventListener('status', (event) => {
    try {
      const status: TrainingMetric = JSON.parse((event as MessageEvent).data);
      onMetric(status);
      // Close connection when training ends
      if (status.status === 'completed' || status.status === 'error') {
        eventSource.close();
      }
    } catch (error) {
      console.error('Failed to parse status:', error);
      if (onError) onError(error as Error);
    }
  });

  eventSource.onerror = (error) => {
    console.error('EventSource error:', error);
    if (onError) onError(new Error('Connection to training stream failed'));
    eventSource.close();
  };

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}
