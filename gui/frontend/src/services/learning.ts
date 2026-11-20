import axios from 'axios'
import type { MetricPoint } from '../types'

export const LEARNING_EXPERIMENT_ID = 'small-model-learning'
export const RESIDUAL_EXPERIMENT_ID = 'residual-connections'
export type LearningExperimentId = typeof LEARNING_EXPERIMENT_ID | typeof RESIDUAL_EXPERIMENT_ID
export type ResidualVariant = 'with-residuals' | 'without-residuals'
export function isLearningExperimentId(value: unknown): value is LearningExperimentId {
  return value === LEARNING_EXPERIMENT_ID || value === RESIDUAL_EXPERIMENT_ID
}
const BASE = '/api/learning'

export type LearningDevice = 'cpu' | 'cuda'
export interface LearningSample { prompt: string; continuation: string }
export interface LearningObservation {
  metrics: { val_loss?: number | null; val_language_loss?: number | null; val_perplexity?: number | null }
  samples: LearningSample[]
  parameters?: number
  initial_weights_sha256?: string
}
export interface LearningDescriptor {
  id: string
  title: string
  description: string
  default_steps: number
  min_steps: number
  max_steps: number
  seed: number
  device: LearningDevice
  default_device: LearningDevice
  available_devices: LearningDevice[]
  hardware?: { cpu_name?: string; cuda_available?: boolean; cuda_device_name?: string | null }
  concepts: { id: string; title: string; url: string }[]
  dataset: { name: string; description: string; train_documents: number; validation_documents: number; license: string; synthetic: boolean }
  model: { d_model: number; n_layers: number; n_heads: number; n_kv_heads: number; max_seq_len: number; vocab_size: number }
  prompts: string[]
  measurement_notes: string[]
  variants?: { id: ResidualVariant; title: string; description: string }[]
  default_variant?: ResidualVariant
}
export interface LearningRun {
  run_id: string
  experiment_id: string
  status: string
  phase: string
  current_step: number
  max_steps: number
  created_at: string
  error?: string | null
  configuration?: {
    steps?: number; seed?: number; device?: LearningDevice; variant?: ResidualVariant
    hardware?: { device?: string; device_name?: string; dtype?: string; torch_version?: string; cuda_version?: string | null; threads?: number }
    model?: Partial<LearningDescriptor['model']>
    training?: { batch_size?: number; gradient_accumulation_steps?: number; [key: string]: unknown }
    dataset?: LearningDescriptor['dataset']
    evaluation?: { prompts?: string[]; [key: string]: unknown }
  }
  metrics: (MetricPoint & { tokens_seen?: number; elapsed_seconds?: number; gradient_norm?: number | null })[]
  before?: LearningObservation | null
  after?: LearningObservation | null
  results?: { before: LearningObservation; after: LearningObservation; step: number; tokens_seen: number } | null
  logs: { message: string; level?: string; timestamp?: number }[]
  tokens_seen?: number | null
  tokens_measured_step?: number | null
  elapsed_seconds?: number | null
}

export interface LearningPreset {
  format: 'llm-lab-learning'
  version: 1
  experiment_id: LearningExperimentId
  steps: number
  seed: number
  variant?: ResidualVariant
}

export function parseLearningPreset(source: string): LearningPreset {
  let value: unknown
  try { value = JSON.parse(source.replace(/^\uFEFF/, '')) } catch { throw new Error('The recipe must contain valid JSON.') }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('This is not an LLM-Lab learning recipe.')
  const data = value as Record<string, unknown>
  const allowed = ['format', 'version', 'experiment_id', 'steps', 'seed', 'variant']
  if (Object.keys(data).some(key => !allowed.includes(key)) || data.format !== 'llm-lab-learning' || data.version !== 1 || !isLearningExperimentId(data.experiment_id)) {
    throw new Error('Use a version 1 learning recipe from Understanding AI.')
  }
  if (data.variant !== undefined && (data.experiment_id !== RESIDUAL_EXPERIMENT_ID || !['with-residuals', 'without-residuals'].includes(String(data.variant)))) throw new Error('The residual variant must be with-residuals or without-residuals.')
  if (!Number.isInteger(data.steps) || (data.steps as number) < 20 || (data.steps as number) > 300) throw new Error('The recipe needs between 20 and 300 optimizer updates.')
  if (!Number.isInteger(data.seed) || (data.seed as number) < 0 || (data.seed as number) > 2147483647) throw new Error('The recipe needs an integer seed between 0 and 2147483647.')
  return data as unknown as LearningPreset
}

export async function getLearningExperiment(experimentId: LearningExperimentId = LEARNING_EXPERIMENT_ID): Promise<LearningDescriptor> {
  return (await axios.get(`${BASE}/experiments/${experimentId}`)).data
}
export async function listLearningRuns(experimentId: LearningExperimentId = LEARNING_EXPERIMENT_ID): Promise<LearningRun[]> {
  return (await axios.get(`${BASE}/runs`, { params: { experiment_id: experimentId } })).data.runs
}
export async function getLearningRun(id: string): Promise<LearningRun> {
  return (await axios.get(`${BASE}/runs/${encodeURIComponent(id)}`)).data
}
export async function startLearningRun(settings: { steps: number; seed: number; device: LearningDevice; variant?: ResidualVariant }, experimentId: LearningExperimentId = LEARNING_EXPERIMENT_ID): Promise<{ run_id: string }> {
  return (await axios.post(`${BASE}/experiments/${experimentId}/runs`, settings)).data
}
export async function stopLearningRun(id: string): Promise<{ run_id: string; status: string }> {
  return (await axios.post(`${BASE}/runs/${encodeURIComponent(id)}/stop`)).data
}

export interface LearningComparison {
  run_a: LearningRun
  run_b: LearningRun
  configuration_differences: { field: string; a: unknown; b: unknown }[]
  comparability: {
    comparable: boolean
    checks: { id: string; label: string; match: boolean | null; detail: string }[]
    warnings: string[]
  }
  metrics: {
    validation_cross_entropy: { a: number | null; b: number | null; delta_b_minus_a: number | null }
    validation_perplexity: { a: number | null; b: number | null; delta_b_minus_a: number | null }
  }
}
export async function compareLearningRuns(runA: string, runB: string): Promise<LearningComparison> {
  return (await axios.get(`${BASE}/comparisons`, { params: { run_a: runA, run_b: runB } })).data
}
