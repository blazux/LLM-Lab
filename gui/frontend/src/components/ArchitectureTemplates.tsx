import { useEffect, useState } from 'react'
import type { ModelConfig } from '../types'

interface Family {
  id: string
  name: string
  description: string
  source: string
  scope?: string
  status?: string
  sizes: Record<string, { d_model: number; n_layers: number; n_heads: number; n_kv_heads: number; d_ff: number; max_seq_len: number }>
}
const field = 'rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100'

export default function ArchitectureTemplates({ onApply }: { onApply: (patch: Partial<ModelConfig>) => void }) {
  const [families, setFamilies] = useState<Family[]>([])
  const [family, setFamily] = useState('llama3')
  const [size, setSize] = useState('nano')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/assembly/templates', { signal: controller.signal }).then(async r => {
      if (!r.ok) throw Error('Could not load architecture templates.')
      const data = await r.json(); if (!Array.isArray(data.templates)) throw Error('Invalid architecture catalog.'); setFamilies(data.templates)
    }).catch(e => { if (!controller.signal.aborted) setError(e.message) })
    return () => controller.abort()
  }, [])
  const selected = families.find(f => f.id === family)
  const dimensions = selected?.sizes[size]
  const apply = async () => {
    setBusy(true); setError('')
    try {
      const response = await fetch('/api/assembly/templates/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ family, size }) })
      const data = await response.json()
      if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Could not build template.')
      onApply(data)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not build template.') }
    finally { setBusy(false) }
  }
  return <section className="rounded-xl border border-slate-700 bg-slate-900/70 p-4 text-slate-100">
    <h2 className="text-sm font-semibold">Start from a known architecture</h2>
    <p className="mt-2 text-xs leading-5 text-slate-400">Build a fresh, scaled architecture with random weights. Keep your tokenizer, then edit any layer. These are versioned starting points, not pretrained checkpoints.</p>
    <div className="mt-3 flex flex-wrap items-end gap-3">
      <label className="flex flex-col gap-1 text-xs text-slate-400">Architecture template<select className={field} value={family} onChange={e => setFamily(e.target.value)} disabled={busy}>{families.map(f => <option key={f.id} value={f.id}>{f.name}</option>)}</select></label>
      <label className="flex flex-col gap-1 text-xs text-slate-400">Starting size<select className={field} value={size} onChange={e => setSize(e.target.value)} disabled={busy}>{Object.keys(selected?.sizes ?? {}).map(s => <option key={s} value={s}>{s[0].toUpperCase() + s.slice(1)}</option>)}</select></label>
      <button className={field + ' border-sky-700 text-sky-200 disabled:opacity-40'} disabled={!selected || busy} onClick={apply}>{busy ? 'Building…' : 'Build editable template'}</button>
    </div>
    {selected && <p className="mt-3 text-xs leading-5 text-slate-400">{selected.description} · <a href={selected.source} target="_blank" rel="noreferrer" className="text-sky-400 underline">Reference</a></p>}
    {selected?.status && <p className="mt-2 text-xs text-amber-200">{selected.status}</p>}
    {selected?.scope && <p className="mt-2 text-xs text-amber-200">{selected.scope}</p>}
    {dimensions && <p className="mt-1 font-mono text-[11px] text-slate-400">d_model {dimensions.d_model} · {dimensions.n_layers} layers · {dimensions.n_heads} Q / {dimensions.n_kv_heads} KV heads · FFN {dimensions.d_ff} · context {dimensions.max_seq_len}</p>}
    <p className="mt-2 text-[11px] text-slate-500">Applying a template replaces the graph. Undo restores your previous model. Parameter count depends on your tokenizer vocabulary.</p>
    {error && <p role="alert" className="mt-2 text-xs text-rose-300">{error}</p>}
  </section>
}
