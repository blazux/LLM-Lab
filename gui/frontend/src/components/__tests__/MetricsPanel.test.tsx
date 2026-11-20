import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import MetricsPanel from '../MetricsPanel'
import { mergeMetricEvent } from '../../utils/metrics'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  LineChart: ({ children, data }: { children: ReactNode; data: unknown }) => <div data-testid="chart" data-points={JSON.stringify(data)}>{children}</div>,
  Line: ({ name }: { name: string }) => <span>{name}</span>,
  YAxis: ({ scale }: { scale: string }) => <span data-testid="axis" data-scale={scale} />,
  XAxis: () => null, Tooltip: () => null, CartesianGrid: () => null, Legend: () => null,
}))

describe('Training and validation metrics', () => {
  const train = { type: 'metrics', step: 100, loss: 3.5, perplexity: 30.25, lr: .0003 }
  const evaluation = { type: 'eval_metrics', step: 100, eval_loss: 3.8, eval_perplexity: 42.75 }
  it('keeps training and evaluation values when they arrive at the same step in either order', () => {
    for (const events of [[train, evaluation], [evaluation, train]]) {
      const result = events.reduce((history, event) => mergeMetricEvent(history, event), [] as Parameters<typeof mergeMetricEvent>[0])
      expect(result).toHaveLength(1)
      expect(result[0]).toMatchObject({ step: 100, loss: 3.5, perplexity: 30.25, eval_loss: 3.8, eval_perplexity: 42.75, lr: .0003 })
      const replacement = mergeMetricEvent(result, { ...evaluation, eval_perplexity: 43 })
      expect(replacement[0]).toMatchObject({ perplexity: 30.25, eval_perplexity: 43, loss: 3.5 })
    }
  })
  it('never substitutes exp(loss) for an absent reported perplexity', () => {
    const history = mergeMetricEvent([], { type: 'metrics', step: 1, loss: 2 })
    expect(history[0].perplexity).toBeUndefined()
    render(<MetricsPanel metrics={history} />)
    expect(screen.getByText('Waiting for reported perplexity…')).toBeTruthy()
  })
  it('rejects nonfinite and nonpositive perplexity without losing a valid loss', () => {
    for (const bad of [null, '42', Infinity, NaN, 0, -2]) {
      const result = mergeMetricEvent([], { ...train, perplexity: bad })
      expect(result[0]).toMatchObject({ loss: 3.5 })
      expect(result[0].perplexity).toBeUndefined()
    }
    expect(mergeMetricEvent([], { ...train, step: NaN })).toEqual([])
  })
  it('shows reported train/validation perplexity, associated steps, and a logarithmic scale control', () => {
    const history = mergeMetricEvent(mergeMetricEvent([], train), { ...evaluation, step: 200 })
    render(<MetricsPanel metrics={history} />)
    expect(screen.getByText('30.25')).toBeTruthy()
    expect(screen.getByText('42.75')).toBeTruthy()
    expect(screen.getByText('Step 100')).toBeTruthy()
    expect(screen.getByText('Step 200')).toBeTruthy()
    expect(screen.getByText(/auxiliary objectives/)).toBeTruthy()
    expect(screen.getAllByTestId('axis').some(axis => axis.getAttribute('data-scale') === 'log')).toBe(true)
    fireEvent.click(screen.getByRole('checkbox', { name: 'Logarithmic perplexity scale' }))
    expect(screen.getAllByTestId('axis').some(axis => axis.getAttribute('data-scale') === 'log')).toBe(false)
  })
  it('labels language cross-entropy and auxiliary losses separately for current runs', () => {
    const history = mergeMetricEvent([], { ...train, language_loss: 3, auxiliary_loss: .5 })
    render(<MetricsPanel metrics={history} metricDefinition="language_cross_entropy_v2" />)
    expect(screen.getByText('Training cross-entropy')).toBeTruthy()
    expect(screen.getByText('Training objective')).toBeTruthy()
    expect(screen.getByText('Training auxiliary loss')).toBeTruthy()
    expect(screen.getByText(/Auxiliary objectives and z-loss are excluded/)).toBeTruthy()
    expect(screen.queryByText(/Legacy metrics/)).toBeNull()
  })
  it('can show a validation-only series before any training point arrives', () => {
    const history = mergeMetricEvent([], evaluation)
    render(<MetricsPanel metrics={history} />)
    expect(screen.getByText('42.75')).toBeTruthy()
    expect(screen.getByRole('region', { name: 'Reported perplexity' })).toBeTruthy()
  })
  it('keeps only language prediction error visible in the guided view and reveals other measurements on request', async () => {
    const history = [{ step: 1, loss: 3, language_loss: 3, eval_loss: 3.2, perplexity: 20, eval_perplexity: 25,
      auxiliary_loss: 0, eval_auxiliary_loss: 0, lr: .001, tokens_per_second: 1000 }]
    render(<MetricsPanel metrics={history} metricDefinition="language_cross_entropy_v2" simpleLanguageView />)
    expect(screen.getAllByTestId('chart')).toHaveLength(1)
    expect(screen.getByRole('region', { name: 'Next-token prediction error' })).toBeTruthy()
    expect(screen.getByText('Training cross-entropy')).toBeTruthy()
    expect(screen.getByText('Validation cross-entropy')).toBeTruthy()
    expect(screen.queryByText('Training objective')).toBeNull()
    expect(screen.queryByText('Training perplexity')).toBeNull()
    const summary = screen.getByText('More training measurements')
    expect(summary.closest('details')?.open).toBe(false)
    fireEvent.click(summary)
    await screen.findAllByText('Training perplexity')
    expect(summary.closest('details')?.open).toBe(true)
    expect(screen.getByRole('region', { name: 'Learning rate' })).toBeTruthy()
    expect(screen.getByRole('region', { name: 'Throughput' })).toBeTruthy()
    expect(screen.queryByRole('region', { name: 'Auxiliary objectives' })).toBeNull()
    expect(screen.queryByText('Training objective')).toBeNull()
    fireEvent.click(summary)
    await waitFor(() => expect(screen.queryAllByText('Training perplexity')).toHaveLength(0))
  })
  it('retains nonzero auxiliary measurements inside the guided details without restoring a duplicate objective curve', async () => {
    render(<MetricsPanel metrics={[{ step: 1, loss: 3.5, language_loss: 3, auxiliary_loss: .5 }]} metricDefinition="language_cross_entropy_v2" simpleLanguageView />)
    expect(screen.queryByText('Training auxiliary loss')).toBeNull()
    fireEvent.click(screen.getByText('More training measurements'))
    await screen.findByText('Training auxiliary loss')
    expect(screen.queryByText('Training objective')).toBeNull()
  })
})
