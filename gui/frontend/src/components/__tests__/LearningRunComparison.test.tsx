import type { ReactNode } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import LearningRunComparison from '../LearningRunComparison'
import * as api from '../../services/learning'
import type { LearningComparison, LearningRun } from '../../services/learning'

vi.mock('../../services/learning', async original => ({ ...await original<typeof api>(), compareLearningRuns: vi.fn() }))
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  LineChart: ({ children, data }: { children: ReactNode; data: unknown }) => <div data-testid="comparison-chart" data-points={JSON.stringify(data)}>{children}</div>,
  Line: ({ name }: { name: string }) => <span>{name}</span>,
  XAxis: () => null, YAxis: () => null, Tooltip: () => null, CartesianGrid: () => null, Legend: () => null,
}))
const first: LearningRun = { run_id: 'a', experiment_id: 'residual-connections', status: 'completed', phase: 'completed',
  current_step: 120, max_steps: 120, created_at: '2026-11-19T12:00:00Z', configuration: { variant: 'with-residuals' },
  metrics: [{ step: 1, language_loss: 5, gradient_norm: 1.5 }, { step: 2, language_loss: 4 }, { step: 2, eval_loss: 4.1 }],
  after: { metrics: { val_language_loss: 4.1 }, samples: [{ prompt: 'A small fox', continuation: ' runs.' }] }, logs: [] }
const second: LearningRun = { ...first, run_id: 'b', configuration: { variant: 'without-residuals' },
  metrics: [{ step: 1, language_loss: 6, gradient_norm: .8 }, { step: 2, eval_loss: 4.2 }],
  after: { metrics: { val_language_loss: 4.2 }, samples: [{ prompt: 'A small fox', continuation: ' walks.' }] } }
const pair: LearningComparison = { run_a: first, run_b: second,
  configuration_differences: [{ field: 'variant', a: 'with-residuals', b: 'without-residuals' }],
  comparability: { comparable: true, checks: [{ id: 'initial_weights', label: 'Initial weights', match: true, detail: 'The saved initial parameter hashes match.' }], warnings: [] },
  metrics: { validation_cross_entropy: { a: 4.1, b: 4.2, delta_b_minus_a: .1 }, validation_perplexity: { a: null, b: null, delta_b_minus_a: null } } }
const props = { runs: [first, second], runA: 'a', runB: 'b', onRunAChange: vi.fn(), onRunBChange: vi.fn() }

beforeEach(() => { vi.clearAllMocks(); vi.mocked(api.compareLearningRuns).mockResolvedValue(pair) })

describe('Recorded A/B comparison', () => {
  it('waits for two distinct recorded runs and keeps selection under caller control', () => {
    const view = render(<LearningRunComparison {...props} runB="" />)
    expect(api.compareLearningRuns).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText('Run B'), { target: { value: 'b' } })
    expect(props.onRunBChange).toHaveBeenCalledWith('b')
    view.rerender(<LearningRunComparison {...props} runB="a" />)
    expect(screen.getByText('Choose two different runs for an A/B comparison.')).toBeTruthy()
    expect(api.compareLearningRuns).not.toHaveBeenCalled()
  })
  it('overlays measured CE and pre-clipping gradients and compares actual continuations', async () => {
    render(<LearningRunComparison {...props} />)
    await screen.findByText('The recorded comparison checks match.')
    expect(api.compareLearningRuns).toHaveBeenCalledWith('a', 'b')
    expect(screen.getByRole('region', { name: 'Training and validation prediction error' })).toBeTruthy()
    expect(screen.getByRole('region', { name: 'Gradient norm before clipping' })).toBeTruthy()
    const points = JSON.parse(screen.getAllByTestId('comparison-chart')[0].getAttribute('data-points')!)
    expect(points).toEqual([{ step: 1, aTrain: 5, aGradient: 1.5, bTrain: 6, bGradient: .8 }, { step: 2, aTrain: 4, aValidation: 4.1, bValidation: 4.2 }])
    expect(screen.getByText('runs.')).toBeTruthy()
    expect(screen.getByText('walks.')).toBeTruthy()
    expect(screen.getByText(/not the clipped norm or a quality score/)).toBeTruthy()
    const checks = screen.getByText(/Recorded comparison checks · 1 match/)
    expect(checks.closest('details')?.open).toBe(false)
    fireEvent.click(checks)
    expect(screen.getByText('Initial weights')).toBeTruthy()
    fireEvent.click(screen.getByText('Configuration differences (1)'))
    expect(screen.getByRole('rowheader', { name: 'Residual connections' })).toBeTruthy()
  })
  it('warns about mismatched or missing conditions without hiding measured outcomes', async () => {
    vi.mocked(api.compareLearningRuns).mockResolvedValue({ ...pair, comparability: { comparable: false,
      checks: [{ id: 'data', label: 'Data', match: false, detail: 'The held-out corpus differs.' }, { id: 'hardware', label: 'Hardware', match: null, detail: 'Hardware was not saved.' }],
      warnings: ['A validation difference cannot be attributed only to the architecture.'] } })
    render(<LearningRunComparison {...props} />)
    await screen.findByText('Some comparison conditions differ or were not recorded.')
    expect(screen.getByText('The held-out corpus differs.')).toBeTruthy()
    expect(screen.getByText('Hardware was not saved.')).toBeTruthy()
    expect(screen.getByText(/cannot be attributed only/)).toBeTruthy()
    expect(screen.getByRole('region', { name: 'Training and validation prediction error' })).toBeTruthy()
  })
  it('presents graph differences by variant name with a short signature in the optional details', async () => {
    vi.mocked(api.compareLearningRuns).mockResolvedValue({ ...pair, configuration_differences: [{ field: 'model.assembly_sha256', a: 'a'.repeat(64), b: 'b'.repeat(64) }] })
    render(<LearningRunComparison {...props} />)
    await screen.findByText('The recorded comparison checks match.')
    fireEvent.click(screen.getByText('Configuration differences (1)'))
    expect(screen.getByRole('rowheader', { name: 'Graph connections' })).toBeTruthy()
    expect(screen.getByText('With residuals · signature aaaaaaaaaaaa…')).toBeTruthy()
    expect(screen.queryByText('a'.repeat(64))).toBeNull()
  })
  it('leaves absent gradients and scalar measurements absent rather than filling zeros', async () => {
    vi.mocked(api.compareLearningRuns).mockResolvedValue({ ...pair,
      run_a: { ...first, metrics: [{ step: 1, language_loss: 5 }] }, run_b: { ...second, metrics: [{ step: 1, eval_loss: 6 }] },
      metrics: { validation_cross_entropy: { a: null, b: null, delta_b_minus_a: null }, validation_perplexity: { a: null, b: null, delta_b_minus_a: null } } })
    render(<LearningRunComparison {...props} />)
    await screen.findByText('Gradient norms were not recorded for these runs.')
    expect(screen.queryByRole('region', { name: 'Gradient norm before clipping' })).toBeNull()
    expect(screen.getAllByText('Not recorded')).toHaveLength(4)
    const points = JSON.parse(screen.getByTestId('comparison-chart').getAttribute('data-points')!)
    expect(points).toEqual([{ step: 1, aTrain: 5, bValidation: 6 }])
  })
  it('ignores a late response for a previously selected pair', async () => {
    let finish!: (value: LearningComparison) => void
    vi.mocked(api.compareLearningRuns).mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    const view = render(<LearningRunComparison {...props} />)
    view.rerender(<LearningRunComparison {...props} runB="c" />)
    await screen.findByText('The recorded comparison checks match.')
    await act(async () => finish({ ...pair, comparability: { ...pair.comparability, warnings: ['Stale response must not appear.'] } }))
    expect(screen.queryByText('Stale response must not appear.')).toBeNull()
    expect(api.compareLearningRuns).toHaveBeenLastCalledWith('a', 'c')
  })
  it('shows a comparison read failure and allows a deliberate retry', async () => {
    vi.mocked(api.compareLearningRuns).mockRejectedValueOnce(new Error('Comparison unavailable.'))
    render(<LearningRunComparison {...props} />)
    await screen.findByRole('alert')
    fireEvent.click(screen.getByRole('button', { name: 'Retry comparison' }))
    await screen.findByText('The recorded comparison checks match.')
    await waitFor(() => expect(api.compareLearningRuns).toHaveBeenCalledTimes(2))
  })
})
