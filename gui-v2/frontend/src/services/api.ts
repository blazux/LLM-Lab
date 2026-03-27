import axios from 'axios'
import type { TrainingConfig, SFTConfig, TrainingStatus, GenerateRequest, GenerateResponse } from '../types'

const BASE = '/api'

const client = axios.create({ baseURL: BASE })

// ─── Training ─────────────────────────────────────────────────────────────────

export async function startPretraining(payload: Record<string, unknown>): Promise<void> {
  await client.post('/training/start', payload)
}

export async function startSFT(payload: Record<string, unknown>): Promise<void> {
  await client.post('/training/sft/start', payload)
}

export async function startRLHF(payload: Record<string, unknown>): Promise<void> {
  await client.post('/training/rlhf/start', payload)
}

export async function stopTraining(): Promise<void> {
  await client.post('/training/stop')
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const res = await client.get('/training/status')
  return res.data
}

// ─── Inference ────────────────────────────────────────────────────────────────

export async function listCheckpoints(): Promise<string[]> {
  const res = await client.get('/inference/checkpoints')
  return res.data
}

export async function generateText(req: GenerateRequest): Promise<GenerateResponse> {
  const res = await client.post('/inference/generate', req)
  return res.data
}

// ─── Health ───────────────────────────────────────────────────────────────────

export async function healthCheck(): Promise<boolean> {
  try {
    await client.get('/health')
    return true
  } catch {
    return false
  }
}

// ─── SSE Metrics Stream ───────────────────────────────────────────────────────

export function createMetricsStream(onMessage: (data: Record<string, unknown>) => void, onError?: () => void): EventSource {
  const es = new EventSource('/api/training/metrics/stream')
  es.onmessage = (e) => {
    try {
      const parsed = JSON.parse(e.data)
      onMessage(parsed)
    } catch {
      // ignore parse errors
    }
  }
  es.onerror = () => {
    if (onError) onError()
  }
  return es
}

// ─── Payload builders ─────────────────────────────────────────────────────────

/** Convert empty strings to null for Optional[int/float] backend fields. */
function sanitizeModelConfig(cfg: Record<string, unknown>): Record<string, unknown> {
  const optionalIntFields = ['d_ff', 'sliding_window', 'd_latent', 'd_rope_latent']
  const result = { ...cfg }
  for (const field of optionalIntFields) {
    if (result[field] === '' || result[field] === undefined) {
      result[field] = null
    }
  }
  return result
}

export function buildPretrainPayload(
  modelConfig: Record<string, unknown>,
  trainingConfig: TrainingConfig
): Record<string, unknown> {
  return {
    model_cfg: sanitizeModelConfig(modelConfig),
    training_cfg: {
      optimizer: trainingConfig.optimizer,
      lr: trainingConfig.lr,
      weight_decay: trainingConfig.weight_decay,
      grad_clip: trainingConfig.grad_clip,
      batch_size: trainingConfig.batch_size,
      gradient_accumulation_steps: trainingConfig.gradient_accumulation_steps,
      max_steps: trainingConfig.max_steps,
      warmup_steps: trainingConfig.warmup_steps,
      scheduler: trainingConfig.scheduler,
      eval_every: trainingConfig.eval_every,
      eval_steps: trainingConfig.eval_steps,
      loss_fn: trainingConfig.loss_fn,
      save_best_only: trainingConfig.save_best_only,
      datasets: trainingConfig.datasets,
      output_dir: trainingConfig.output_dir,
      // Optimizer-specific
      adamw_beta1: trainingConfig.adamw_beta1,
      adamw_beta2: trainingConfig.adamw_beta2,
      adamw_eps: trainingConfig.adamw_eps,
      muon_momentum: trainingConfig.muon_momentum,
      muon_nesterov: trainingConfig.muon_nesterov,
      lion_beta1: trainingConfig.lion_beta1,
      lion_beta2: trainingConfig.lion_beta2,
      sophia_beta1: trainingConfig.sophia_beta1,
      sophia_beta2: trainingConfig.sophia_beta2,
      sophia_rho: trainingConfig.sophia_rho,
      // Scheduler-specific
      adaptive_window: trainingConfig.adaptive_window,
      adaptive_patience: trainingConfig.adaptive_patience,
      adaptive_increase_factor: trainingConfig.adaptive_increase_factor,
      adaptive_decrease_factor: trainingConfig.adaptive_decrease_factor,
      adaptive_min_lr: trainingConfig.adaptive_min_lr,
      adaptive_threshold: trainingConfig.adaptive_threshold,
      // Loss-specific
      maxis_n_candidates: trainingConfig.maxis_n_candidates,
      maxis_low_rank_dim: trainingConfig.maxis_low_rank_dim,
      maxis_chunk_size: trainingConfig.maxis_chunk_size,
      maxis_aux_weight: trainingConfig.maxis_aux_weight,
    },
  }
}

// SFTRequest is a flat object — no model_cfg nesting (model is loaded from checkpoint)
export function buildSFTPayload(
  _modelConfig: Record<string, unknown>,
  trainingConfig: SFTConfig
): Record<string, unknown> {
  return {
    policy_checkpoint: trainingConfig.policy_checkpoint,
    datasets: trainingConfig.datasets,
    optimizer: trainingConfig.optimizer,
    learning_rate: trainingConfig.learning_rate,
    weight_decay: trainingConfig.weight_decay,
    batch_size: trainingConfig.batch_size,
    gradient_accumulation_steps: trainingConfig.gradient_accumulation_steps,
    max_steps: trainingConfig.max_steps,
    warmup_steps: trainingConfig.warmup_steps,
    scheduler: trainingConfig.scheduler,
    max_grad_norm: trainingConfig.max_grad_norm,
    log_every: trainingConfig.log_every,
    save_every: trainingConfig.save_every,
    eval_every: trainingConfig.eval_every,
    eval_steps: trainingConfig.eval_steps,
    save_best_only: trainingConfig.save_best_only,
    output_dir: trainingConfig.output_dir,
    dropout: trainingConfig.dropout_override ?? null,
    use_lora: trainingConfig.use_lora,
    lora_preset: trainingConfig.lora_preset,
    lora_target_modules: trainingConfig.lora_target_modules ?? null,
    lora_r: trainingConfig.lora_r,
    lora_alpha: trainingConfig.lora_alpha,
    lora_dropout: trainingConfig.lora_dropout,
    // Optimizer-specific
    adamw_beta1: trainingConfig.adamw_beta1,
    adamw_beta2: trainingConfig.adamw_beta2,
    adamw_eps: trainingConfig.adamw_eps,
    muon_momentum: trainingConfig.muon_momentum,
    muon_nesterov: trainingConfig.muon_nesterov,
    lion_beta1: trainingConfig.lion_beta1,
    lion_beta2: trainingConfig.lion_beta2,
    sophia_beta1: trainingConfig.sophia_beta1,
    sophia_beta2: trainingConfig.sophia_beta2,
    sophia_rho: trainingConfig.sophia_rho,
    // Scheduler-specific
    adaptive_window: trainingConfig.adaptive_window,
    adaptive_patience: trainingConfig.adaptive_patience,
    adaptive_increase_factor: trainingConfig.adaptive_increase_factor,
    adaptive_decrease_factor: trainingConfig.adaptive_decrease_factor,
    adaptive_min_lr: trainingConfig.adaptive_min_lr,
    adaptive_threshold: trainingConfig.adaptive_threshold,
  }
}
