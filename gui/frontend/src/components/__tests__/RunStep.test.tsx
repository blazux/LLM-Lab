import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import RunStep from '../steps/RunStep'
import { defaultTransformerConfig, defaultTrainingConfig, type Project, type TrainingStatus } from '../../types'
import * as api from '../../services/api'

const mock = vi.hoisted(() => ({ updateProject: vi.fn(), listeners: new Map<string, (event: Record<string, unknown>) => void>(), close: vi.fn() }))
vi.mock('../../context/ProjectContext', () => ({ useProjects: () => ({ updateProject: mock.updateProject }) }))
vi.mock('../MetricsPanel', () => ({ default: ({ metrics }: { metrics: unknown[] }) => <pre data-testid="metrics">{JSON.stringify(metrics)}</pre> }))
vi.mock('../../services/api', async (original) => ({
  ...await original<typeof api>(), getTrainingStatus: vi.fn(), listRuns: vi.fn(), getRun: vi.fn(), resumeRun: vi.fn(),
  createMetricsStream: vi.fn((listener, _error, runId) => { mock.listeners.set(runId, listener); return { close: mock.close } }),
}))
const training = defaultTrainingConfig()
const model = defaultTransformerConfig()
const project: Project = { id: 'p', name: 'My project', type: 'pretrain', createdAt: '', updatedAt: '', currentStep: 0,
  lastRunId: 'saved-a', modelConfig: model, trainingConfig: { ...training, optimizer: 'lion', max_steps: 999 } }
const saved = { run_id: 'saved-a', project_id: 'p', status: 'completed', is_training: false, current_step: 10, max_steps: 10,
  training_type: 'pretraining', metric_definition: 'language_cross_entropy_v2', output_dir: '/saved/a',
  model_config: { ...model, model_architecture: 'transformer' }, training_config: { ...training, optimizer: 'adamw', max_steps: 10 } } as unknown as TrainingStatus

beforeEach(() => {
  vi.clearAllMocks(); mock.listeners.clear()
  HTMLElement.prototype.scrollIntoView = vi.fn()
  vi.mocked(api.getTrainingStatus).mockResolvedValue({ ...saved, run_id: 'other-active', is_training: true })
  vi.mocked(api.listRuns).mockResolvedValue([saved, { ...saved, run_id: 'saved-b', output_dir: '/saved/b' }])
  vi.mocked(api.getRun).mockImplementation(async id => ({ ...saved, run_id: id }))
})
describe('Recorded run monitoring', () => {
  it('uses saved configuration and replays only selected-run events, without duplicates', async () => {
    const view = render(<RunStep project={project} />)
    await screen.findByText('ADAMW')
    expect(screen.queryByText('LION')).toBeNull()
    expect(screen.getByRole('button', { name: /Start new training/ }).hasAttribute('disabled')).toBe(true)
    const emit = mock.listeners.get('saved-a')!
    act(() => {
      emit({ type: 'metrics', run_id: 'wrong', event_id: 1, step: 1, loss: 99 })
      emit({ type: 'metrics', run_id: 'saved-a', event_id: 2, step: 1, loss: 2, language_loss: 1, perplexity: Math.E })
      emit({ type: 'metrics', run_id: 'saved-a', event_id: 2, step: 1, loss: 99 })
    })
    expect(JSON.parse(screen.getByTestId('metrics').textContent!)).toEqual([{ step: 1, loss: 2, language_loss: 1, perplexity: Math.E }])
    expect(screen.getAllByText(/step=1/)).toHaveLength(1)
    view.unmount()
    expect(mock.close).toHaveBeenCalled()
    render(<RunStep project={project} />)
    act(() => mock.listeners.get('saved-a')!({ type: 'metrics', run_id: 'saved-a', event_id: 2, step: 1, loss: 2 }))
    expect(JSON.parse(screen.getByTestId('metrics').textContent!)).toEqual([{ step: 1, loss: 2 }])
  })
  it('switches to the chosen run, clears old curves, and resumes its saved checkpoint', async () => {
    vi.mocked(api.getTrainingStatus).mockResolvedValue(saved)
    vi.mocked(api.resumeRun).mockResolvedValue({ run_id: 'resumed' })
    render(<RunStep project={project} />)
    await screen.findByText('ADAMW')
    act(() => mock.listeners.get('saved-a')!({ type: 'metrics', run_id: 'saved-a', step: 1, loss: 2 }))
    fireEvent.change(screen.getByLabelText('Recorded run'), { target: { value: 'saved-b' } })
    await waitFor(() => expect(api.getRun).toHaveBeenCalledWith('saved-b'))
    expect(screen.getByTestId('metrics').textContent).toBe('[]')
    expect(mock.updateProject).toHaveBeenCalledWith('p', { lastRunId: 'saved-b' })
    await screen.findByRole('button', { name: /Resume saved checkpoint/ })
    fireEvent.change(screen.getByLabelText('Additional optimizer steps'), { target: { value: '20' } })
    fireEvent.click(screen.getByRole('button', { name: /Resume saved checkpoint/ }))
    await waitFor(() => expect(api.resumeRun).toHaveBeenCalledWith('saved-b', 20))
    expect(mock.updateProject).toHaveBeenCalledWith('p', { lastRunId: 'resumed' })
  })
})
