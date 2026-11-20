import axios from 'axios'
import type { TrainingConfig, SFTConfig, RLHFConfig, TrainingStatus, GenerateRequest, GenerateResponse } from '../types'

const BASE = '/api'

const client = axios.create({ baseURL: BASE })

// ─── Training ─────────────────────────────────────────────────────────────────

export async function startPretraining(payload: Record<string, unknown>): Promise<{ run_id: string }> {
  return (await client.post('/training/start', payload)).data
}

export async function startSFT(payload: Record<string, unknown>): Promise<{ run_id: string }> {
  return (await client.post('/training/sft/start', payload)).data
}

export async function startRLHF(payload: Record<string, unknown>): Promise<{ run_id: string }> {
  return (await client.post('/training/rlhf/start', payload)).data
}

export async function stopTraining(runId?: string): Promise<void> {
  await client.post('/training/stop', null, { params: { run_id: runId } })
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const res = await client.get('/training/status')
  return res.data
}

export async function listRuns(): Promise<TrainingStatus[]> {
  return (await client.get('/training/runs')).data.runs
}

export async function getRun(runId: string): Promise<TrainingStatus> {
  return (await client.get(`/training/runs/${encodeURIComponent(runId)}`)).data
}

export async function importRun(outputDir: string, projectId: string): Promise<{ run_id: string }> {
  return (await client.post('/training/runs/import', { output_dir: outputDir, project_id: projectId })).data
}

export async function resumeRun(runId: string, additionalSteps: number): Promise<{ run_id: string }> {
  return (await client.post(`/training/runs/${encodeURIComponent(runId)}/resume`, { additional_steps: additionalSteps })).data
}

// ─── Inference ────────────────────────────────────────────────────────────────

export async function listCheckpoints(runId?: string): Promise<CheckpointInfo[]> {
  const res = await client.get('/inference/checkpoints', { params: { run_id: runId || undefined } })
  return res.data.checkpoints
}

/** Extract a readable message from an axios/FastAPI error. */
export function errMessage(err: unknown): string {
  if (typeof err === 'object' && err !== null && 'response' in err) {
    const detail = (err as { response?: { data?: { detail?: unknown } } }).response?.data?.detail
    if (typeof detail === 'string') return detail
    if (detail !== undefined) return JSON.stringify(detail)
  }
  return err instanceof Error ? err.message : 'Unknown error'
}

export async function generateText(req: GenerateRequest): Promise<GenerateResponse> {
  const res = await client.post('/inference/generate', req)
  return res.data
}

// ─── Export / Push to HuggingFace ────────────────────────────────────────────

export interface CheckpointInfo {
  name: string
  path: string
  size_mb: number
  modified: number
  step?: number | string
  type?: string
  architecture?: string
  attention_type?: string
  has_lora?: boolean
  error?: string
}

export interface FormatDetection {
  format: string
  reason: string
  vllm_compatible: boolean
  model_type: string
  architectures: string[]
  checkpoint_info: Record<string, unknown>
}

export interface ExportRequestPayload {
  checkpoint_path: string
  output_dir?: string
  repo_id?: string
  hf_token?: string
  private?: boolean
  model_name?: string
  model_description?: string
  license_type?: string
  author?: string
}

export interface ExportResult {
  success: boolean
  message: string
  format?: string
  output_path?: string
  hub_url?: string
  files?: string[]
}

export async function listExportCheckpoints(runId?: string): Promise<CheckpointInfo[]> {
  const res = await client.get('/export/checkpoints', { params: { run_id: runId || undefined } })
  return res.data.checkpoints
}

export async function detectExportFormat(checkpointPath: string): Promise<FormatDetection> {
  const res = await client.get('/export/detect-format', { params: { checkpoint_path: checkpointPath } })
  return res.data
}

export async function exportLocal(payload: ExportRequestPayload): Promise<ExportResult> {
  const res = await client.post('/export/local', payload)
  return res.data
}

export async function exportToHub(payload: ExportRequestPayload): Promise<ExportResult> {
  const res = await client.post('/export/hub', payload)
  return res.data
}

// ─── Merge LoRA ───────────────────────────────────────────────────────────────

export interface AdapterInfo {
  name: string
  path: string
  rank: number | string
  alpha: number | string
  target_modules: string[]
  modified: number
}

export interface MergeResult {
  success: boolean
  message: string
  output_path?: string
}

export async function listMergeAvailable(): Promise<{ checkpoints: CheckpointInfo[]; adapters: AdapterInfo[] }> {
  const res = await client.get('/merge/available')
  return res.data
}

export async function mergeLora(payload: {
  adapter_path: string
  base_checkpoint_path: string
  output_path: string
  input_type: 'adapter_folder' | 'full_checkpoint'
}): Promise<MergeResult> {
  const res = await client.post('/merge/lora', payload)
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

export function createMetricsStream(onMessage: (data: Record<string, unknown>) => void, onError?: () => void, runId?: string): EventSource {
  const es = new EventSource(`/api/training/metrics/stream${runId ? `?run_id=${encodeURIComponent(runId)}` : ''}`)
  const forward = (e: MessageEvent) => {
    try {
      onMessage(JSON.parse(e.data))
    } catch {
      // ignore parse errors
    }
  }
  es.onmessage = forward
  // The backend also emits named SSE events — forward them all
  es.addEventListener('snapshot', forward as EventListener)
  es.addEventListener('status', forward as EventListener)
  es.onerror = () => {
    if (onError) onError()
  }
  return es
}

// ─── Payload builders ─────────────────────────────────────────────────────────

/** Adapt the frontend config to the backend ModelConfig schema:
 *  - rename `architecture` (frontend) to `model_architecture` (backend)
 *  - convert empty strings to null for Optional[int/float] backend fields. */
export function sanitizeModelConfig(cfg: Record<string, unknown>): Record<string, unknown> {
  const optionalIntFields = ['d_ff', 'sliding_window', 'd_latent', 'd_rope_latent', 'nope_every']
  const { architecture, ...result } = cfg as Record<string, unknown> & { architecture?: string }
  result.model_architecture = architecture ?? 'transformer'
  for (const field of optionalIntFields) {
    if (result[field] === '' || result[field] === undefined) {
      result[field] = null
    }
  }
  if (result.hybrid_pattern === '') result.hybrid_pattern = null
  return result
}

export function buildPretrainPayload(
  modelConfig: Record<string, unknown>,
  trainingConfig: TrainingConfig
): Record<string, unknown> {
  return {
    model_cfg: sanitizeModelConfig(modelConfig),
    output_dir: trainingConfig.output_dir,
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
      wsd_decay_fraction: trainingConfig.wsd_decay_fraction,
      z_loss_weight: trainingConfig.z_loss_weight,
      use_compile: trainingConfig.use_compile,
      use_doc_masks: trainingConfig.use_doc_masks,
      eval_every: trainingConfig.eval_every,
      eval_steps: trainingConfig.eval_steps,
      loss_fn: trainingConfig.loss_fn,
      save_best_only: trainingConfig.save_best_only,
      datasets: trainingConfig.datasets,
      validation_datasets: trainingConfig.validation_datasets ?? null,
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
    loss_fn: trainingConfig.loss_fn,
    maxis_n_candidates: trainingConfig.maxis_n_candidates,
    maxis_low_rank_dim: trainingConfig.maxis_low_rank_dim,
    maxis_chunk_size: trainingConfig.maxis_chunk_size,
    maxis_aux_weight: trainingConfig.maxis_aux_weight,
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

// RLHFRequest is a flat object matching the backend pydantic model
export function buildRLHFPayload(cfg: RLHFConfig): Record<string, unknown> {
  return {
    algorithm: cfg.algorithm,
    policy_checkpoint: cfg.policy_checkpoint,
    datasets: cfg.datasets,
    optimizer: cfg.optimizer,
    learning_rate: cfg.learning_rate,
    weight_decay: cfg.weight_decay,
    batch_size: cfg.batch_size,
    mini_batch_size: cfg.mini_batch_size,
    gradient_accumulation_steps: cfg.gradient_accumulation_steps,
    max_steps: cfg.max_steps,
    max_grad_norm: cfg.max_grad_norm,
    log_every: cfg.log_every,
    save_every: cfg.save_every,
    eval_every: cfg.eval_every,
    output_dir: cfg.output_dir,
    // Generation (rollouts)
    max_new_tokens: cfg.max_new_tokens,
    temperature: cfg.temperature,
    top_k: cfg.top_k,
    top_p: cfg.top_p,
    // LoRA
    use_lora: cfg.use_lora,
    lora_preset: cfg.lora_preset,
    lora_target_modules: cfg.lora_target_modules ?? null,
    lora_r: cfg.lora_r,
    lora_alpha: cfg.lora_alpha,
    lora_dropout: cfg.lora_dropout,
    // Optimizer-specific
    adamw_beta1: cfg.adamw_beta1,
    adamw_beta2: cfg.adamw_beta2,
    adamw_eps: cfg.adamw_eps,
    muon_momentum: cfg.muon_momentum,
    muon_nesterov: cfg.muon_nesterov,
    lion_beta1: cfg.lion_beta1,
    lion_beta2: cfg.lion_beta2,
    sophia_beta1: cfg.sophia_beta1,
    sophia_beta2: cfg.sophia_beta2,
    sophia_rho: cfg.sophia_rho,
    // Algorithm-specific
    reward_model_name: cfg.reward_model_name || null,
    reference_checkpoint: cfg.reference_checkpoint || null,
    ppo_epochs: cfg.ppo_epochs,
    clip_range: cfg.clip_range,
    gamma: cfg.gamma,
    gae_lambda: cfg.gae_lambda,
    vf_coef: cfg.vf_coef,
    group_size: cfg.group_size,
    grpo_temperature: cfg.grpo_temperature,
    grpo_clip_higher: cfg.grpo_clip_higher === '' ? null : cfg.grpo_clip_higher,
    grpo_dynamic_sampling: cfg.grpo_dynamic_sampling,
    grpo_importance_level: cfg.grpo_importance_level,
  }
}
