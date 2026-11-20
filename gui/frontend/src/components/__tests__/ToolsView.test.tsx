import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ToolsView from '../ToolsView'
import * as api from '../../services/api'
vi.mock('../../services/api', async original => ({
  ...await original<typeof api>(), listRuns: vi.fn(), listExportCheckpoints: vi.fn(), listCheckpoints: vi.fn(),
  detectExportFormat: vi.fn(), exportToHub: vi.fn(),
}))
const checkpoint = { name: 'run-a / final_model.pt', path: '/custom/run-a/final_model.pt', size_mb: 1, modified: 1 }
beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api.listRuns).mockResolvedValue([{ run_id: 'run-a', status: 'completed', current_step: 4, training_type: 'pretraining' } as never])
  vi.mocked(api.listExportCheckpoints).mockResolvedValue([checkpoint])
  vi.mocked(api.listCheckpoints).mockResolvedValue([checkpoint])
  vi.mocked(api.detectExportFormat).mockResolvedValue({ format: 'custom', reason: 'Executable graph', vllm_compatible: false, model_type: 'assembly', architectures: [], checkpoint_info: {} })
  vi.mocked(api.exportToHub).mockResolvedValue({ success: true, message: 'Uploaded privately' })
})
describe('Run-scoped model tools', () => {
  it('exposes labeled controls, filters a recorded run, and pushes privately with backend authentication', async () => {
    render(<ToolsView onBack={() => {}} />)
    await screen.findByRole('option', { name: /run-a · pretraining/ })
    fireEvent.change(screen.getByLabelText('Tools recorded run'), { target: { value: 'run-a' } })
    await waitFor(() => expect(api.listExportCheckpoints).toHaveBeenCalledWith('run-a'))
    await screen.findByRole('option', { name: /run-a \/ final_model/ })
    fireEvent.change(screen.getByLabelText('Checkpoint'), { target: { value: checkpoint.path } })
    fireEvent.change(screen.getByLabelText('Repository'), { target: { value: 'fixture/my-model' } })
    expect(screen.getByRole('switch', { name: 'Private repository' }).getAttribute('aria-checked')).toBe('true')
    fireEvent.click(screen.getByRole('button', { name: 'Push to Hub' }))
    await waitFor(() => expect(api.exportToHub).toHaveBeenCalledWith(expect.objectContaining({ checkpoint_path: checkpoint.path, repo_id: 'fixture/my-model', private: true, hf_token: '' })))
    expect(await screen.findByText('Uploaded privately')).toBeTruthy()
  })
  it('keeps the recorded-run filter when changing to Playground', async () => {
    render(<ToolsView onBack={() => {}} />)
    await screen.findByRole('option', { name: /run-a · pretraining/ })
    fireEvent.change(screen.getByLabelText('Tools recorded run'), { target: { value: 'run-a' } })
    fireEvent.click(screen.getByRole('button', { name: 'Playground' }))
    await waitFor(() => expect(api.listCheckpoints).toHaveBeenCalledWith('run-a'))
  })
})
