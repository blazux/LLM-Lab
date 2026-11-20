import type { MetricPoint } from '../types'

export function finiteMetric(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

/** Preserve the server's metrics; never manufacture perplexity from a loss. */
export function mergeMetricEvent(history: MetricPoint[], event: Record<string, unknown>): MetricPoint[] {
  const step = finiteMetric(event.step)
  if (step === undefined || step < 0 || !Number.isInteger(step)) return history
  const fields = event.type === 'eval_metrics'
    ? { eval_loss: event.eval_loss, eval_perplexity: event.eval_perplexity, eval_language_loss: event.eval_language_loss, eval_auxiliary_loss: event.eval_auxiliary_loss, eval_objective_loss: event.eval_objective_loss }
    : event.type === 'metrics'
      ? { loss: event.loss, language_loss: event.language_loss, auxiliary_loss: event.auxiliary_loss, perplexity: event.perplexity, lr: event.lr, tokens_per_second: event.tokens_per_second }
      : {}
  const point: MetricPoint = { step }
  for (const [key, raw] of Object.entries(fields)) {
    const value = finiteMetric(raw)
    if (value !== undefined && (!key.includes('perplexity') || value > 0)) {
      Object.assign(point, { [key]: value })
    }
  }
  if (Object.keys(point).length === 1) return history
  const previous = history.find(m => m.step === step)
  return [...history.filter(m => m.step !== step), { ...previous, ...point }].sort((a, b) => a.step - b.step)
}
