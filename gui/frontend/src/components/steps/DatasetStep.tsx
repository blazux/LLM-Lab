import DatasetGuide from './DatasetGuide'
import DatasetSources from './DatasetSources'
import { Plus, Trash2, Check, Star } from 'lucide-react'
import type { TrainingConfig, DatasetEntry, ProjectType, RLHFConfig } from '../../types'

// ─── Curated catalog ──────────────────────────────────────────────────────────
// Only datasets whose schema the backend loaders actually parse:
//  - pretrain: needs a `text` column (streamed, EOS-packed)
//  - sft: messages / ShareGPT / Alpaca / prompt+completion / OpenOrca formats
//  - rlhf: chosen/rejected pairs (DPO) or prompts (PPO/GRPO)

interface CatalogEntry {
  name: string
  split: string
  subset?: string
  size: string
  desc: string
  recommended?: boolean
}

const SFT_CATALOG: CatalogEntry[] = [
  {
    name: 'HuggingFaceTB/smol-smoltalk', split: 'train', size: '~500k conv', recommended: true,
    desc: 'The SmolLM2 SFT mix, built specifically for small models. Best default here.',
  },
  {
    name: 'HuggingFaceH4/ultrachat_200k', split: 'train_sft', size: '200k conv',
    desc: 'Filtered multi-turn conversations — the Zephyr recipe. Note the train_sft split.',
  },
  {
    name: 'HuggingFaceH4/no_robots', split: 'train', size: '10k conv',
    desc: '100% human-written demonstrations. Small but gold-standard quality — great as a final polish.',
  },
  {
    name: 'allenai/tulu-3-sft-mixture', split: 'train', size: '940k conv',
    desc: 'AllenAI’s state-of-the-art open SFT mixture (Tülu 3).',
  },
  {
    name: 'teknium/OpenHermes-2.5', split: 'train', size: '1M conv',
    desc: 'Large curated instruction mix (ShareGPT format), widely used by open fine-tunes.',
  },
  {
    name: 'yahma/alpaca-cleaned', split: 'train', size: '52k pairs',
    desc: 'The cleaned classic Alpaca instruction set — simple single-turn baseline.',
  },
]

const DPO_CATALOG: CatalogEntry[] = [
  {
    name: 'Anthropic/hh-rlhf', split: 'train', size: '161k pairs', recommended: true,
    desc: 'Helpful & harmless chosen/rejected pairs — the gold-standard preference set.',
  },
  {
    name: 'argilla/ultrafeedback-binarized-preferences-cleaned', split: 'train', size: '61k pairs',
    desc: 'Cleaned UltraFeedback preferences — the pairs behind Zephyr-beta.',
  },
  {
    name: 'Intel/orca_dpo_pairs', split: 'train', size: '13k pairs',
    desc: 'High-quality reasoning preference pairs distilled from Orca.',
  },
]

const PROMPT_CATALOG: CatalogEntry[] = [
  {
    name: 'Anthropic/hh-rlhf', split: 'train', size: '161k prompts', recommended: true,
    desc: 'General helpfulness prompts — responses are generated live and scored by the reward model.',
  },
]

const EXPLAINER: Record<string, string> = {
  pretrain:
    'Your corpus defines the language, style and topics the model can learn. Documents are tokenized, terminated with an EOS token, and packed into fixed-length sequences. With several datasets, the weight biases the sampling mix — modern recipes upweight high-quality sources (math, textbooks, code) especially near the end of training.',
  sft:
    'SFT teaches the pretrained model to behave like an assistant. Here diversity and quality beat volume: Choose examples that demonstrate the behavior you want, and inspect their quality. Formats are auto-detected (messages, ShareGPT, Alpaca…) and rendered through the chat template; only assistant tokens contribute to the loss.',
  rlhf_dpo:
    'DPO needs preference pairs: for each prompt, a chosen and a rejected response. The model learns to widen the gap between them relative to a frozen reference. Quality of the preference judgments is everything — noisy pairs teach noise.',
  rlhf_gen:
    'PPO and GRPO only need prompts: the model generates its own responses at every step, and the reward model scores them. Pick prompts that match what you want to improve — the reward model must be competent on that domain, since the policy will exploit its blind spots.',
}

// ─── DatasetStep ──────────────────────────────────────────────────────────────

interface Props {
  projectType: ProjectType
  config: TrainingConfig
  onChange: (cfg: TrainingConfig) => void
  sequenceLength?: number
}

export default function DatasetStep({ projectType, config, onChange, sequenceLength }: Props) {
  const isRLHF = projectType === 'rlhf'
  const algo = isRLHF ? ((config as RLHFConfig).algorithm ?? 'grpo') : null
  const showWeight = !isRLHF && config.datasets.length > 1

  const catalog =
    projectType === 'pretrain' ? [] :
    projectType === 'sft' ? SFT_CATALOG :
    algo === 'dpo' ? DPO_CATALOG : PROMPT_CATALOG

  const explainer =
    projectType === 'rlhf'
      ? (algo === 'dpo' ? EXPLAINER.rlhf_dpo : EXPLAINER.rlhf_gen)
      : EXPLAINER[projectType]

  function addDataset(entry?: CatalogEntry) {
    const ds: DatasetEntry = entry
      ? { name: entry.name, split: entry.split, ...(entry.subset ? { subset: entry.subset } : {}) }
      : { name: '', split: 'train' }
    // replace a single still-empty row instead of stacking under it
    const base = config.datasets.length === 1 && config.datasets[0].name === ''
      ? [] : config.datasets
    if (entry && base.some((d) => d.name === entry.name)) return
    onChange({ ...config, datasets: [...base, ds] })
  }

  function removeDataset(idx: number) {
    onChange({
      ...config,
      datasets: config.datasets.filter((_, i) => i !== idx),
    })
  }

  function updateDataset(idx: number, field: keyof DatasetEntry, value: string | number | undefined) {
    onChange({
      ...config,
      datasets: config.datasets.map((d, i) => (i === idx ? { ...d, [field]: value } : d)),
    })
  }

  const isAdded = (name: string) => config.datasets.some((d) => d.name === name)

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h3 className="text-base font-semibold text-slate-200 mb-1">Training Datasets</h3>
        <p className="text-sm text-slate-500">
          Choose Hugging Face datasets or local files. Streaming fetches data as needed and can still use local cache space.
        </p>
      </div>

      <p className="text-xs text-slate-400 leading-relaxed bg-slate-800/40 border border-slate-700/60 rounded-lg px-3 py-2">
        {explainer}
      </p>

      {projectType === 'pretrain' && <DatasetGuide config={config} onChange={onChange} sequenceLength={sequenceLength} />}

      {projectType === 'pretrain' && <DatasetSources title="Training sources" sources={config.datasets} onChange={datasets => onChange({ ...config, datasets })} />}
      {/* Dataset list */}
      {projectType !== 'pretrain' && <div className="space-y-3">
        {config.datasets.map((ds, idx) => (
          <div key={idx} className="flex gap-2 items-start bg-slate-800/50 border border-slate-700 rounded-xl p-3">
            <div className="flex-none w-5 h-5 mt-2 rounded flex items-center justify-center bg-accent/10 text-accent text-xs font-bold">
              {idx + 1}
            </div>
            <div className={`flex-1 grid gap-2 ${showWeight ? 'grid-cols-[3fr_1.4fr_1fr_0.8fr]' : 'grid-cols-[3fr_1.4fr_1fr]'}`}>
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Dataset name</label>
                <input
                  type="text"
                  value={ds.name}
                  onChange={(e) => updateDataset(idx, 'name', e.target.value)}
                  placeholder="HuggingFaceFW/fineweb-edu"
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-hidden focus:border-accent placeholder-slate-600"
                />
              </div>
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Subset <span className="text-slate-700">(optional)</span></label>
                <input
                  type="text"
                  value={ds.subset ?? ''}
                  onChange={(e) => updateDataset(idx, 'subset', e.target.value || undefined)}
                  placeholder="—"
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-hidden focus:border-accent placeholder-slate-600"
                />
              </div>
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Split</label>
                <input
                  type="text"
                  value={ds.split}
                  onChange={(e) => updateDataset(idx, 'split', e.target.value)}
                  placeholder="train"
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-hidden focus:border-accent placeholder-slate-600"
                />
              </div>
              {showWeight && (
                <div>
                  <label className="text-xs text-slate-500 mb-1 block">Weight</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    value={ds.weight ?? 1}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value)
                      updateDataset(idx, 'weight', Number.isFinite(v) ? v : undefined)
                    }}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-hidden focus:border-accent"
                  />
                </div>
              )}

            </div>
            <button
              onClick={() => removeDataset(idx)}
              disabled={config.datasets.length === 1}
              className="mt-2 p-1.5 text-slate-600 hover:text-red-400 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>}

      <button
        onClick={() => addDataset()}
        className="flex items-center gap-2 text-sm text-accent hover:text-accent/80 transition-colors"
      >
        <Plus size={14} />
        Add custom dataset
      </button>

      {projectType === 'pretrain' && <section className="space-y-3">
        <h3 className="text-sm font-semibold text-slate-200">Validation: measure learning on unseen documents</h3>
        <p className="text-xs text-slate-400">A falling training loss may just mean memorization. Validation loss checks a held-out source. Keep it fixed when comparing models.</p>
        <label className="flex gap-2 text-sm text-slate-300"><input type="checkbox" checked={config.validation_datasets != null} onChange={e => onChange({ ...config, validation_datasets: e.target.checked ? [{ name: '', split: 'validation' }] : undefined })} />Use separate validation documents</label>
        {config.validation_datasets != null
          ? <DatasetSources title="Validation sources" sources={config.validation_datasets} onChange={validation_datasets => onChange({ ...config, validation_datasets })} />
          : <p className="text-xs text-amber-200/80">Legacy fallback: reserves 10% of packed sequences. Parts of a long document can occur in both sets; use a guided preset or separate sources for stronger validation.</p>}
        {config.validation_datasets != null && <button className="text-sm text-accent" onClick={() => onChange({ ...config, validation_datasets: [...config.validation_datasets!, { name: '', split: 'validation' }] })}>Add validation source</button>}
      </section>}
      {/* Curated catalog */}
      {projectType !== 'pretrain' && <div>
        <div className="flex items-center gap-3 mb-3">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
            Dataset suggestions
          </span>
          <div className="flex-1 h-px bg-slate-800" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {catalog.map((entry) => {
            const added = isAdded(entry.name)
            return (
              <button
                key={entry.name}
                onClick={() => addDataset(entry)}
                disabled={added}
                className={`text-left rounded-xl border p-3 transition-colors ${
                  added
                    ? 'bg-accent/5 border-accent/30 cursor-default'
                    : 'bg-slate-800/40 border-slate-700/60 hover:border-accent/50 hover:bg-slate-800/70'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  {entry.recommended && <Star size={11} className="text-amber-400 flex-none" fill="currentColor" />}
                  <span className="text-xs font-medium text-slate-200 font-mono truncate">{entry.name}</span>
                  <span className="ml-auto flex-none text-[10px] text-slate-500 bg-slate-900/70 rounded-full px-2 py-0.5">
                    {entry.size}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 leading-relaxed">{entry.desc}</p>
                <div className={`mt-1.5 flex items-center gap-1 text-[11px] ${added ? 'text-accent' : 'text-slate-600'}`}>
                  {added ? (<><Check size={11} /> Added</>) : (<><Plus size={11} /> Add{entry.split !== 'train' ? ` (split: ${entry.split})` : ''}</>)}
                </div>
              </button>
            )
          })}
        </div>
        {isRLHF && algo !== 'dpo' && (
          <p className="text-xs text-slate-600 mt-2">
            Any prompt dataset works for {String(algo).toUpperCase()} — responses are generated during training, so
            what matters is that the prompts cover the behaviors you want to reward.
          </p>
        )}
      </div>}

      {/* Output directory */}
      <div className="pt-2">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Output</span>
          <div className="flex-1 h-px bg-slate-800" />
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-slate-400 w-28 flex-none">output_dir</label>
          <input
            type="text"
            value={config.output_dir}
            onChange={(e) => onChange({ ...config, output_dir: e.target.value })}
            placeholder="/app/data"
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-hidden focus:border-accent placeholder-slate-600"
          />
        </div>
        <p className="text-xs text-slate-600 mt-1.5 ml-32">Checkpoints and logs will be saved here</p>
      </div>

      {/* Summary */}
      <div className="bg-slate-800/30 rounded-xl p-4 text-xs text-slate-500 space-y-1 border border-slate-800">
        <div className="flex justify-between">
          <span>Number of datasets</span>
          <span className="text-slate-300">{config.datasets.length}</span>
        </div>
        <div className="flex justify-between">
          <span>Output directory</span>
          <span className="text-slate-300 font-mono">{config.output_dir || '(not set)'}</span>
        </div>
      </div>
    </div>
  )
}
