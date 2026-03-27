import { Plus, Trash2 } from 'lucide-react'
import type { TrainingConfig, DatasetEntry } from '../../types'

interface Props {
  config: TrainingConfig
  onChange: (cfg: TrainingConfig) => void
}

export default function DatasetStep({ config, onChange }: Props) {
  function addDataset() {
    onChange({
      ...config,
      datasets: [...config.datasets, { name: '', split: 'train' }],
    })
  }

  function removeDataset(idx: number) {
    onChange({
      ...config,
      datasets: config.datasets.filter((_, i) => i !== idx),
    })
  }

  function updateDataset(idx: number, field: keyof DatasetEntry, value: string) {
    onChange({
      ...config,
      datasets: config.datasets.map((d, i) => (i === idx ? { ...d, [field]: value } : d)),
    })
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h3 className="text-base font-semibold text-slate-200 mb-1">Training Datasets</h3>
        <p className="text-sm text-slate-500">
          Add one or more HuggingFace datasets. They are streamed during training — no local download required.
        </p>
      </div>

      {/* Dataset list */}
      <div className="space-y-3">
        {config.datasets.map((ds, idx) => (
          <div key={idx} className="flex gap-2 items-start bg-slate-800/50 border border-slate-700 rounded-xl p-3">
            <div className="flex-none w-5 h-5 mt-2 rounded flex items-center justify-center bg-accent/10 text-accent text-xs font-bold">
              {idx + 1}
            </div>
            <div className="flex-1 grid grid-cols-2 gap-2">
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Dataset name</label>
                <input
                  type="text"
                  value={ds.name}
                  onChange={(e) => updateDataset(idx, 'name', e.target.value)}
                  placeholder="HuggingFaceFW/fineweb-edu"
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-accent placeholder-slate-600"
                />
              </div>
              <div>
                <label className="text-xs text-slate-500 mb-1 block">Split</label>
                <input
                  type="text"
                  value={ds.split}
                  onChange={(e) => updateDataset(idx, 'split', e.target.value)}
                  placeholder="train"
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-accent placeholder-slate-600"
                />
              </div>
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
      </div>

      <button
        onClick={addDataset}
        className="flex items-center gap-2 text-sm text-accent hover:text-accent/80 transition-colors"
      >
        <Plus size={14} />
        Add dataset
      </button>

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
            className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-accent placeholder-slate-600"
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
