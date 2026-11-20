import { useId } from 'react'
import type { DatasetEntry } from '../../types'

interface Props { title: string; sources: DatasetEntry[]; onChange: (sources: DatasetEntry[]) => void }
const inputClass = 'mt-1 w-full rounded-lg border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-100'

export default function DatasetSources({ title, sources, onChange }: Props) {
  const id = useId()
  function update(index: number, field: keyof DatasetEntry, value: string | number | undefined) {
    onChange(sources.map((source, i) => i === index ? { ...source, [field]: value } : source))
  }
  return <fieldset className="space-y-3"><legend className="mb-2 text-sm font-medium text-slate-300">{title}</legend>
    {sources.map((source, index) => {
      const textInput = (field: keyof DatasetEntry, label: string, placeholder: string) => <label htmlFor={`${id}-${index}-${field}`} className="block text-xs text-slate-400">{label}
        <input id={`${id}-${index}-${field}`} className={inputClass} value={source[field] ?? ''} placeholder={placeholder} onChange={e => update(index, field, e.target.value || (field === 'name' || field === 'split' ? '' : undefined))} />
      </label>
      const numberInput = (field: 'skip' | 'take' | 'weight', label: string, min: number, placeholder: string) => <label htmlFor={`${id}-${index}-${field}`} className="block text-xs text-slate-400">{label}
        <input id={`${id}-${index}-${field}`} type="number" min={min} step={field === 'weight' ? 'any' : 1} className={inputClass} value={source[field] ?? ''} placeholder={placeholder} onChange={e => update(index, field, e.target.value === '' ? undefined : Number(e.target.value))} />
      </label>
      return <div key={index} className="space-y-3 rounded-xl border border-slate-700 bg-slate-800/40 p-3">
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-[3fr_1fr_1fr]">
          {textInput('name', 'Dataset name / local file builder', 'roneneldan/TinyStories')}
          {textInput('subset', 'Subset (optional)', 'sample-10BT')}
          {textInput('split', 'Split', 'train')}
        </div>
        <p className="break-all text-xs text-slate-400">{source.split || 'train'} · skip {source.skip ?? 0} documents · {source.take ? `take ${source.take.toLocaleString()}` : 'no document limit'} · text column: {source.text_field || 'text'}</p>
        <details><summary className="cursor-pointer text-xs text-accent">Source options: document range, revision, text column and local files</summary>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            {numberInput('skip', 'Skip documents', 0, '0')}
            {numberInput('take', 'Take documents (optional)', 1, 'Unlimited')}
            {numberInput('weight', 'Document sampling weight', 0.001, '1')}
            {textInput('text_field', 'Text column', 'text')}
            <div className="sm:col-span-2">{textInput('revision', 'Hub revision (commit recommended)', 'Commit hash, tag or branch')}</div>
            <div className="sm:col-span-3">{textInput('data_files', 'Local file (optional)', '/path/to/corpus.jsonl')}</div>
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-500">Local files: use json, text or parquet as the dataset name. JSONL needs a text field per row; plain text uses one document per line. Skip and Take count rows before tokenization. With multiple sources, weights control document sampling, not exact token percentages; the mixture stops when a source is exhausted, then the trainer restarts it. Pin a commit for repeatable runs.</p>
        </details>
        {sources.length > 1 && <button className="text-xs text-red-400" onClick={() => onChange(sources.filter((_, i) => i !== index))}>Remove source {index + 1}</button>}
      </div>
    })}
  </fieldset>
}
