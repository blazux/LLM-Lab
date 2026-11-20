import { useRef, useState } from 'react'
import { sanitizeModelConfig } from '../services/api'
import type { ModelConfig } from '../types'

export default function AssemblySize({ config, onApply }: { config: ModelConfig; onApply: (c: ModelConfig) => void }) {
  const current = useRef(config)
  current.current = config
  const [width, setWidth] = useState(config.d_model)
  const [layers, setLayers] = useState(config.assembly?.version === 2 ? config.n_layers : Number(config.assembly?.nodes.find(n => n.kind === 'repeat')?.params?.count ?? 2))
  const [context, setContext] = useState(config.max_seq_len)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const apply = async () => {
    const requested = JSON.stringify(config)
    setBusy(true); setError('')
    try {
      const response = await fetch('/api/assembly/resize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model_cfg: sanitizeModelConfig(config as unknown as Record<string, unknown>), width, layers, context }) })
      const data = await response.json()
      if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Could not resize model.')
      if (JSON.stringify(current.current) !== requested) throw Error('The model changed while resizing. Apply the size again to resize your current graph.')
      onApply({ ...config, assembly: data.model_cfg.assembly, d_model: data.model_cfg.d_model, n_layers: data.model_cfg.n_layers, max_seq_len: data.model_cfg.max_seq_len })
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not resize model.') }
    finally { setBusy(false) }
  }
  return <details className="mt-3 rounded-lg border border-slate-700 p-3">
    <summary className="cursor-pointer text-xs text-sky-300">Resize the whole model</summary>
    <p className="mt-2 text-[11px] leading-5 text-slate-400">Adjusts all layer dimensions, FFN sizes and head counts together. Keeps your connections, activation choices and FFN expansion ratios. For modern templates, depth edits rebuild the root graph and keep surviving layer bodies. Custom root-level additions may be replaced. Width-only changes preserve the entire graph.</p>
    {([['Global model dimension', width, setWidth, 2], ['Global layer count', layers, setLayers, 1], ['Global context length', context, setContext, 2]] as const).map(([label, value, setter, min]) => <label key={label} className="mt-2 block text-xs text-slate-400">{label}<input className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2 text-slate-100" type="number" min={min} step={label === 'Global model dimension' ? 2 : 1} value={value} onChange={e => setter(Number(e.target.value))} disabled={busy} /></label>)}
    <button onClick={apply} disabled={busy} className="mt-3 rounded-lg border border-sky-700 px-3 py-2 text-xs text-sky-200 disabled:opacity-40">{busy ? 'Resizing…' : 'Apply global size'}</button>
    {error && <p role="alert" className="mt-2 text-xs text-rose-300">{error}</p>}
  </details>
}
