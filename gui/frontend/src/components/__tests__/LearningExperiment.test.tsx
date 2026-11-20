import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../../App'
import LearningExperiment from '../LearningExperiment'
import * as api from '../../services/learning'
import type { LearningComparison, LearningDescriptor, LearningRun } from '../../services/learning'

vi.mock('../../services/learning', async original => ({
  ...await original<typeof api>(), getLearningExperiment: vi.fn(), getLearningRun: vi.fn(), listLearningRuns: vi.fn(),
  startLearningRun: vi.fn(), stopLearningRun: vi.fn(),
  compareLearningRuns: vi.fn(),
}))
vi.mock('../MetricsPanel', () => ({ default: ({ metrics }: { metrics: unknown[] }) => <pre data-testid="learning-metrics">{JSON.stringify(metrics)}</pre> }))

const descriptor: LearningDescriptor = {
  id: 'small-model-learning', title: 'What does a small model learn?', description: 'A controlled learning task.',
  default_steps: 120, min_steps: 20, max_steps: 300, seed: 42, device: 'cuda', default_device: 'cuda',
  available_devices: ['cpu', 'cuda'], hardware: { cpu_name: 'Test CPU', cuda_available: true, cuda_device_name: 'Test GPU' },
  dataset: { name: 'Toy sentences', description: 'Artificial sentences with distinct training and validation documents.', train_documents: 320, validation_documents: 64, license: 'MIT', synthetic: true },
  model: { d_model: 64, n_layers: 2, n_heads: 4, n_kv_heads: 2, max_seq_len: 64, vocab_size: 512 },
  concepts: [], prompts: ['The little fox'], measurement_notes: [],
}
const recorded: LearningRun = {
  run_id: 'saved-learning', experiment_id: 'small-model-learning', status: 'completed', phase: 'completed',
  current_step: 120, max_steps: 120, created_at: '2026-11-19T12:00:00Z', error: null,
  configuration: { steps: 120, seed: 42, device: 'cpu', model: descriptor.model,
    training: { batch_size: 4, gradient_accumulation_steps: 1 },
    hardware: { device: 'cpu', device_name: 'Recorded CPU', dtype: 'float32', threads: 4 } },
  before: { parameters: 100000, metrics: { val_language_loss: 6, val_perplexity: 403.429 }, samples: [{ prompt: 'The little fox', continuation: ' random tokens' }] },
  after: { metrics: { val_language_loss: 2, val_perplexity: 7.389 }, samples: [{ prompt: 'The little fox', continuation: ' walks in the garden.' }] },
  metrics: [{ step: 1, language_loss: 5, loss: 5 }, { step: 120, eval_loss: 2, eval_perplexity: 7.389 }],
  tokens_seen: 30240, elapsed_seconds: 10,
  logs: [{ message: 'Validation completed.', level: 'info', timestamp: 1 }],
}
const residualDescriptor: LearningDescriptor = { ...descriptor, id: 'residual-connections', title: 'What do residual connections change?',
  default_variant: 'with-residuals', variants: [{ id: 'with-residuals', title: 'With residuals', description: 'Two bypass additions.' }, { id: 'without-residuals', title: 'Without residuals', description: 'Both additions removed.' }] }
const comparison: LearningComparison = { run_a: recorded, run_b: { ...recorded, run_id: 'saved-b' },
  configuration_differences: [], comparability: { comparable: true, checks: [{ id: 'seed', label: 'Seed', match: true, detail: 'Seed 42.' }], warnings: [] },
  metrics: { validation_cross_entropy: { a: 2, b: 2, delta_b_minus_a: 0 }, validation_perplexity: { a: 7.389, b: 7.389, delta_b_minus_a: 0 } } }

beforeEach(() => {
  vi.clearAllMocks()
  window.history.replaceState(null, '', '/')
  localStorage.clear()
  vi.mocked(api.getLearningExperiment).mockResolvedValue(descriptor)
  vi.mocked(api.listLearningRuns).mockResolvedValue([recorded])
  vi.mocked(api.getLearningRun).mockImplementation(async id => ({ ...recorded, run_id: id }))
  vi.mocked(api.startLearningRun).mockResolvedValue({ run_id: 'new-learning' })
  vi.mocked(api.stopLearningRun).mockResolvedValue({ run_id: 'saved-learning', status: 'stopping' })
  vi.mocked(api.compareLearningRuns).mockResolvedValue(comparison)
})

describe('Guided learning experience', () => {
  it('opens a deep link without starting a run and uses the available preferred device', async () => {
    window.history.replaceState(null, '', '/#experiment=small-model-learning')
    render(<App />)
    await screen.findByRole('option', { name: 'NVIDIA GPU · Test GPU' })
    expect((screen.getByLabelText('Compute device') as HTMLSelectElement).value).toBe('cuda')
    expect(api.startLearningRun).not.toHaveBeenCalled()
    expect(screen.getByRole('heading', { name: 'What does a small model learn?' })).toBeTruthy()
    expect(screen.getByRole('link', { name: /How pretraining works/ }).getAttribute('href')).toBe('https://www.thelastserver.com/understanding-ai/#tech-gpt1')
  })

  it('offers the experiment on the dashboard without creating a project or starting training', async () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: /Open guided experiment/ }))
    await screen.findByRole('option', { name: 'NVIDIA GPU · Test GPU' })
    expect(window.location.hash).toBe('#experiment=small-model-learning')
    expect(api.startLearningRun).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Projects' }))
    expect(screen.getByRole('button', { name: /Open guided experiment/ })).toBeTruthy()
    expect(window.location.hash).toBe('')
    expect(JSON.parse(localStorage.getItem('llmlab_v2_projects') || '[]')).toHaveLength(0)
  })

  it('only starts after an explicit action, sends reviewed settings and preserves the run in the URL', async () => {
    render(<LearningExperiment onBack={vi.fn()} />)
    await screen.findByRole('option', { name: 'NVIDIA GPU · Test GPU' })
    fireEvent.change(screen.getByLabelText('Optimizer updates'), { target: { value: '40' } })
    fireEvent.change(screen.getByLabelText('Random seed'), { target: { value: '123' } })
    fireEvent.change(screen.getByLabelText('Compute device'), { target: { value: 'cpu' } })
    fireEvent.click(screen.getByRole('button', { name: 'Start experiment' }))
    await waitFor(() => expect(api.startLearningRun).toHaveBeenCalledWith({ steps: 40, seed: 123, device: 'cpu' }, 'small-model-learning'))
    await screen.findByText('walks in the garden.', { exact: false })
    expect(window.location.hash).toBe('#experiment=small-model-learning&run=new-learning')
    expect(api.getLearningRun).toHaveBeenCalledWith('new-learning')
  })

  it('reopens saved observations and exact before/after continuations without inventing fresh metrics', async () => {
    window.history.replaceState(null, '', '/#experiment=small-model-learning&run=saved-learning')
    const view = render(<LearningExperiment onBack={vi.fn()} />)
    await screen.findByText('random tokens', { exact: false })
    expect(screen.getByText('walks in the garden.', { exact: false })).toBeTruthy()
    expect(screen.getByText(/Measured on Recorded CPU/)).toBeTruthy()
    expect(JSON.parse(screen.getByTestId('learning-metrics').textContent!)).toEqual(recorded.metrics)
    fireEvent.click(screen.getByText('Saved run log (1 entries)'))
    expect(screen.getByText('Validation completed.')).toBeTruthy()
    view.unmount()
    render(<LearningExperiment onBack={vi.fn()} />)
    await screen.findByText('walks in the garden.', { exact: false })
    expect(api.getLearningRun).toHaveBeenCalledTimes(2)
    expect(api.startLearningRun).not.toHaveBeenCalled()
    expect(JSON.parse(screen.getByTestId('learning-metrics').textContent!)).toHaveLength(2)
  })

  it('selects recorded runs and stops only the selected active run', async () => {
    const active = { ...recorded, status: 'training', phase: 'training', current_step: 20, after: null }
    vi.mocked(api.listLearningRuns).mockResolvedValue([active])
    vi.mocked(api.getLearningRun).mockResolvedValue(active)
    render(<LearningExperiment onBack={vi.fn()} />)
    await screen.findByRole('option', { name: /training.*20\/120/ })
    fireEvent.change(screen.getByLabelText('Recorded experiment'), { target: { value: 'saved-learning' } })
    fireEvent.click(await screen.findByRole('button', { name: 'Stop experiment' }))
    await waitFor(() => expect(api.stopLearningRun).toHaveBeenCalledWith('saved-learning'))
    expect(api.startLearningRun).not.toHaveBeenCalled()
  })

  it('loads the site recipe as settings without starting or overriding the device', async () => {
    render(<LearningExperiment onBack={vi.fn()} />)
    await screen.findByRole('option', { name: 'NVIDIA GPU · Test GPU' })
    const file = new File([JSON.stringify({ format: 'llm-lab-learning', version: 1, experiment_id: 'small-model-learning', steps: 80, seed: 7 })], 'learning.json', { type: 'application/json' })
    fireEvent.change(screen.getByLabelText('Import experiment recipe'), { target: { files: [file] } })
    await screen.findByText(/Recipe loaded/)
    expect((screen.getByLabelText('Optimizer updates') as HTMLInputElement).value).toBe('80')
    expect((screen.getByLabelText('Random seed') as HTMLInputElement).value).toBe('7')
    expect((screen.getByLabelText('Compute device') as HTMLSelectElement).value).toBe('cuda')
    expect(api.startLearningRun).not.toHaveBeenCalled()
  })

  it('rejects recipes containing executable or unknown fields and validates the budget', async () => {
    const preset = { format: 'llm-lab-learning', version: 1, experiment_id: 'small-model-learning', steps: 120, seed: 42 }
    expect(() => api.parseLearningPreset(JSON.stringify({ ...preset, url: 'http://localhost:8000/run' }))).toThrow()
    expect(() => api.parseLearningPreset(JSON.stringify({ ...preset, steps: 301 }))).toThrow()
    expect(() => api.parseLearningPreset(JSON.stringify({ ...preset, seed: 1.5 }))).toThrow()
    expect(() => api.parseLearningPreset(JSON.stringify({ ...preset, experiment_id: 'unknown' }))).toThrow()
    render(<LearningExperiment onBack={vi.fn()} />)
    await screen.findByRole('option', { name: 'NVIDIA GPU · Test GPU' })
    fireEvent.change(screen.getByLabelText('Optimizer updates'), { target: { value: '301' } })
    expect((screen.getByRole('button', { name: 'Start experiment' }) as HTMLButtonElement).disabled).toBe(true)
    expect(api.startLearningRun).not.toHaveBeenCalled()
  })

  it('shows server errors and retains the experience instead of claiming a run started', async () => {
    vi.mocked(api.startLearningRun).mockRejectedValue(new Error('Another training run is active.'))
    render(<LearningExperiment onBack={vi.fn()} />)
    await screen.findByRole('option', { name: 'NVIDIA GPU · Test GPU' })
    fireEvent.click(screen.getByRole('button', { name: 'Start experiment' }))
    expect(await screen.findByRole('alert')).toHaveProperty('textContent', 'Another training run is active.')
    expect(api.getLearningRun).not.toHaveBeenCalled()
    expect(window.location.hash).toBe('')
  })

  it('does not offer unavailable CUDA or show missing measurements as zero', async () => {
    vi.mocked(api.getLearningExperiment).mockResolvedValue({ ...descriptor, default_device: 'cpu', device: 'cpu', available_devices: ['cpu'] })
    vi.mocked(api.getLearningRun).mockResolvedValue({ ...recorded, status: 'interrupted', phase: 'interrupted', before: null, after: null, metrics: [], tokens_seen: null, elapsed_seconds: null, error: 'Server restarted.' })
    window.history.replaceState(null, '', '/#experiment=small-model-learning&run=saved-learning')
    render(<LearningExperiment onBack={vi.fn()} />)
    await screen.findByText('Server restarted.')
    expect(screen.queryByRole('option', { name: /NVIDIA GPU/ })).toBeNull()
    expect(screen.getAllByText('Not measured yet').length).toBeGreaterThan(3)
    expect(screen.queryByTestId('learning-metrics')).toBeNull()
    expect(screen.getByText(/did not complete its full budget/)).toBeTruthy()
  })
  it('opens the residual experience and explicitly starts only the reviewed variant', async () => {
    vi.mocked(api.getLearningExperiment).mockResolvedValue(residualDescriptor)
    vi.mocked(api.listLearningRuns).mockResolvedValue([])
    vi.mocked(api.getLearningRun).mockResolvedValue({ ...recorded, experiment_id: 'residual-connections', configuration: { ...recorded.configuration, variant: 'without-residuals' } })
    window.history.replaceState(null, '', '/#experiment=residual-connections')
    render(<App />)
    await screen.findByRole('option', { name: 'NVIDIA GPU · Test GPU' })
    expect(screen.getByRole('heading', { name: 'What do residual connections change?' })).toBeTruthy()
    expect(api.startLearningRun).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText('Residual connections'), { target: { value: 'without-residuals' } })
    expect(screen.getByRole('img', { name: /without bypass additions/ })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Start experiment' }))
    await waitFor(() => expect(api.startLearningRun).toHaveBeenCalledWith({ steps: 120, seed: 42, device: 'cuda', variant: 'without-residuals' }, 'residual-connections'))
    expect(api.startLearningRun).toHaveBeenCalledTimes(1)
    expect(window.location.hash).toBe('#experiment=residual-connections&run=new-learning')
  })
  it('prepares the other residual variant from saved settings without starting it', async () => {
    const residualRun: LearningRun = { ...recorded, experiment_id: 'residual-connections', configuration: { ...recorded.configuration, steps: 80, seed: 9, device: 'cpu', variant: 'with-residuals' } }
    vi.mocked(api.getLearningExperiment).mockResolvedValue(residualDescriptor)
    vi.mocked(api.getLearningRun).mockResolvedValue(residualRun)
    vi.mocked(api.listLearningRuns).mockResolvedValue([residualRun])
    window.history.replaceState(null, '', '/#experiment=residual-connections&run=saved-learning')
    render(<App />)
    fireEvent.click(await screen.findByRole('button', { name: 'Prepare the other variant from this run' }))
    expect((screen.getByLabelText('Residual connections') as HTMLSelectElement).value).toBe('without-residuals')
    expect((screen.getByLabelText('Optimizer updates') as HTMLInputElement).value).toBe('80')
    expect((screen.getByLabelText('Random seed') as HTMLInputElement).value).toBe('9')
    expect((screen.getByLabelText('Compute device') as HTMLSelectElement).value).toBe('cpu')
    expect(api.startLearningRun).not.toHaveBeenCalled()
  })
  it('imports a residual recipe from the first experience, opens the matching view and never starts it', async () => {
    vi.mocked(api.getLearningExperiment).mockImplementation(async id => id === 'residual-connections' ? residualDescriptor : descriptor)
    window.history.replaceState(null, '', '/#experiment=small-model-learning')
    render(<App />)
    await screen.findByRole('option', { name: 'NVIDIA GPU · Test GPU' })
    const preset = { format: 'llm-lab-learning', version: 1, experiment_id: 'residual-connections', steps: 100, seed: 8, variant: 'without-residuals' }
    fireEvent.change(screen.getByLabelText('Import experiment recipe'), { target: { files: [new File([JSON.stringify(preset)], 'residual.json', { type: 'application/json' })] } })
    await screen.findByRole('heading', { name: 'What do residual connections change?' })
    await waitFor(() => expect((screen.getByLabelText('Optimizer updates') as HTMLInputElement).value).toBe('100'))
    expect((screen.getByLabelText('Random seed') as HTMLInputElement).value).toBe('8')
    expect((screen.getByLabelText('Residual connections') as HTMLSelectElement).value).toBe('without-residuals')
    expect(window.location.hash).toBe('#experiment=residual-connections')
    expect(api.startLearningRun).not.toHaveBeenCalled()
    expect(() => api.parseLearningPreset(JSON.stringify({ ...preset, variant: 'other' }))).toThrow()
    expect(() => api.parseLearningPreset(JSON.stringify({ ...preset, experiment_id: 'small-model-learning' }))).toThrow()
  })
  it('persists A/B selections in the first experience and restores them after reopening', async () => {
    vi.mocked(api.listLearningRuns).mockResolvedValue([recorded, { ...recorded, run_id: 'saved-b' }])
    window.history.replaceState(null, '', '/#experiment=small-model-learning&run=saved-learning')
    const view = render(<App />)
    await screen.findByText('random tokens', { exact: false })
    fireEvent.change(screen.getByLabelText('Run B'), { target: { value: 'saved-b' } })
    await waitFor(() => expect(api.compareLearningRuns).toHaveBeenCalledWith('saved-learning', 'saved-b'))
    expect(window.location.hash).toBe('#experiment=small-model-learning&run=saved-learning&compare=saved-b')
    view.unmount()
    render(<App />)
    await waitFor(() => expect(api.compareLearningRuns).toHaveBeenCalledTimes(2))
    expect((screen.getByLabelText('Run A') as HTMLSelectElement).value).toBe('saved-learning')
    expect((screen.getByLabelText('Run B') as HTMLSelectElement).value).toBe('saved-b')
    expect(api.startLearningRun).not.toHaveBeenCalled()
  })
})
