import { useState } from 'react'
import type { TrainingConfig } from '../../types'
import { PRETRAINING_CORPORA, corpusSources } from '../../content/pretraining-datasets'

interface Props { config: TrainingConfig; onChange: (config: TrainingConfig) => void; sequenceLength?: number }

export default function DatasetGuide({ config, onChange, sequenceLength }: Props) {
  const [choiceId, setChoiceId] = useState('stories')
  const [limited, setLimited] = useState(true)
  const choice = PRETRAINING_CORPORA.find(c => c.id === choiceId)!
  const input = 'rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100'
  return <section aria-label="Pretraining dataset guide" className="space-y-4 rounded-xl border border-accent/30 bg-slate-800/40 p-4">
    <div><h3 className="font-semibold text-slate-200">What would you like your model to learn?</h3>
      <p className="mt-1 text-sm text-slate-400">For a first run, choose TinyStories and a small model. Pretraining learns to continue text; teaching an assistant comes later with SFT.</p></div>
    <label className="block text-xs text-slate-400">Learning goal
      <select aria-label="Learning goal" className={`${input} mt-1 block w-full`} value={choiceId} onChange={e => setChoiceId(e.target.value)}>
        {PRETRAINING_CORPORA.map(c => <option key={c.id} value={c.id}>{c.purpose} — {c.title} ({c.language})</option>)}
      </select>
    </label>
    <div className="space-y-2 text-sm text-slate-300">
      <div className="flex flex-wrap justify-between gap-2"><strong>{choice.title} · {choice.language}</strong>
        <a className="text-accent underline" href={`https://huggingface.co/datasets/${choice.source.name}/blob/${choice.source.revision}/README.md`} target="_blank" rel="noreferrer">Read dataset card</a></div>
      <p>{choice.quality}</p><p className="text-xs text-amber-200/80">Limits: {choice.limitation}</p>
      <p className="text-xs text-slate-400">Published size: {choice.size}. This is not your training budget.</p>
      <p className="text-xs text-slate-400">Dataset license: {choice.license}. The card describes source terms and attribution for sharing your work.</p>
    </div>
    <label className="flex items-start gap-2 text-sm text-slate-300"><input type="checkbox" className="mt-1" checked={limited} onChange={e => setLimited(e.target.checked)} />
      <span>Start with 40,000 training documents<span className="mt-1 block text-xs text-slate-500">A document limit is not a token limit: lengths vary. When the stream ends, training reuses it until Max steps is reached. Streaming fetches and caches data as needed; a shard may be larger than this selection.</span></span></label>
    <p className="text-xs leading-5 text-slate-400">Validation: {choice.validationSplit ? '512 documents from the separate validation split.' : 'reserve the first 512 training documents, then train on later documents. The published test split stays untouched.'} These documents are kept out of training. Row separation does not remove duplicate text across the corpus.</p>
    <button className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white" onClick={() => onChange({ ...config, ...corpusSources(choice, limited ? 40000 : undefined) })}>Use {choice.title} — replace data sources</button>
    <div className="border-t border-slate-700 pt-3 text-xs text-slate-400">
      <p className="mb-2 text-slate-300" aria-live="polite">Current run: {config.max_steps.toLocaleString('en-US')} updates × {config.batch_size} examples × accumulation {config.gradient_accumulation_steps}{sequenceLength && sequenceLength > 1 ? ` × ${sequenceLength - 1} predicted tokens = approximately ${(config.max_steps * config.batch_size * config.gradient_accumulation_steps * (sequenceLength - 1)).toLocaleString('en-US')} tokens.` : '.'} Repeated documents count again.</p>
      <p>For a short first test: 1,000 optimizer updates, batch size 1, accumulation 4, evaluation every 100 updates on up to 20 batches. Runtime and VRAM depend on your model and context length.</p>
      <button className="mt-2 text-accent underline" onClick={() => onChange({ ...config, max_steps: 1000, warmup_steps: 100, batch_size: 1, gradient_accumulation_steps: 4, eval_every: 100, eval_steps: 20 })}>Apply short test training settings</button>
      <p className="mt-2">Next: choose a tokenizer matching the language, check the token budget in Training, then watch validation loss and generated samples. Compare runs with the same tokenizer and held-out corpus.</p>
    </div>
  </section>
}
