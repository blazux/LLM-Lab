import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import { render, screen, fireEvent, within } from '@testing-library/react'
import DatasetStep from '../steps/DatasetStep'
import { defaultTrainingConfig, type TrainingConfig } from '../../types'
import { buildPretrainPayload } from '../../services/api'
import { PRETRAINING_CORPORA } from '../../content/pretraining-datasets'

let latest: TrainingConfig
function Harness({ type = 'pretrain' }: { type?: 'pretrain' | 'sft' }) {
  const [config, setConfig] = useState(defaultTrainingConfig())
  latest = config
  return <DatasetStep projectType={type} config={config} onChange={setConfig} sequenceLength={512} />
}

describe('Guided pretraining sources', () => {
  for (const corpus of PRETRAINING_CORPORA) {
    it(`sends actual ${corpus.title} sources, document ranges and revision to training`, () => {
      render(<Harness />)
      fireEvent.change(screen.getByLabelText('Learning goal'), { target: { value: corpus.id } })
      fireEvent.click(screen.getByRole('button', { name: `Use ${corpus.title} — replace data sources` }))
      const cfg = buildPretrainPayload({}, latest).training_cfg as TrainingConfig
      expect(cfg.datasets).toEqual([{ ...corpus.source, text_field: 'text', skip: corpus.validationSplit ? 0 : 512, take: 40000 }])
      expect(cfg.validation_datasets).toEqual([{ ...corpus.source, text_field: 'text', split: corpus.validationSplit ?? 'train', take: 512 }])
      expect(cfg.datasets[0].revision).toMatch(/^[a-f0-9]{40}$/)
      expect(screen.getByRole('link', { name: 'Read dataset card' }).getAttribute('href')).toContain(corpus.source.revision)
    })
  }
  it('allows editing the selected sources and disabling the training document limit', () => {
    render(<Harness />)
    fireEvent.click(screen.getByRole('checkbox', { name: /Start with 40,000/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Use TinyStories — replace data sources' }))
    expect(latest.datasets[0].take).toBeUndefined()
    const train = within(screen.getByRole('group', { name: 'Training sources' }))
    fireEvent.change(train.getByLabelText('Text column'), { target: { value: 'body' } })
    fireEvent.change(train.getByLabelText('Take documents (optional)'), { target: { value: '1200' } })
    expect(latest.datasets[0]).toMatchObject({ text_field: 'body', take: 1200 })
    expect(latest.validation_datasets![0].text_field).toBe('text')
  })
  it('applies a bounded training budget separately from the corpus choice', () => {
    render(<Harness />)
    const original = latest.datasets
    fireEvent.click(screen.getByRole('button', { name: 'Apply short test training settings' }))
    expect(latest).toMatchObject({ max_steps: 1000, warmup_steps: 100, batch_size: 1, gradient_accumulation_steps: 4, eval_every: 100, eval_steps: 20 })
    expect(latest.datasets).toEqual(original)
    expect(screen.getByText(/approximately 2,044,000 tokens/)).toBeTruthy()
  })
  it('explains the legacy holdout when separate validation is disabled', () => {
    render(<Harness />)
    fireEvent.click(screen.getByRole('button', { name: 'Use TinyStories — replace data sources' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Use separate validation documents' }))
    expect(latest.validation_datasets).toBeUndefined()
    expect(screen.getByText(/Legacy fallback/)).toBeTruthy()
    expect((buildPretrainPayload({}, latest).training_cfg as TrainingConfig).validation_datasets).toBeNull()
  })
  it('keeps pretraining corpora out of the SFT recommendations', () => {
    render(<Harness type="sft" />)
    expect(screen.queryByLabelText('Pretraining dataset guide')).toBeNull()
    expect(screen.getByText('HuggingFaceTB/smol-smoltalk')).toBeTruthy()
  })
})
