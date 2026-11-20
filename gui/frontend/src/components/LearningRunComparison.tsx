import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'
import { compareLearningRuns, type LearningComparison, type LearningRun } from '../services/learning'
import { errMessage } from '../services/api'

interface Props {
  runs: LearningRun[]
  runA: string
  runB: string
  onRunAChange: (id: string) => void
  onRunBChange: (id: string) => void
}
type ComparisonPoint = { step: number; aTrain?: number; aValidation?: number; bTrain?: number; bValidation?: number; aGradient?: number; bGradient?: number }
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const value = (number: number | null | undefined) => finite(number) ? number.toLocaleString('en-US', { maximumFractionDigits: 4 }) : 'Not recorded'
const active = (run: LearningRun) => ['starting', 'preparing', 'training', 'stopping', 'evaluating', 'sampling'].includes(run.status)
const variantName = (run: LearningRun) => run.configuration?.variant === 'with-residuals' ? 'With residuals' : run.configuration?.variant === 'without-residuals' ? 'Without residuals' : run.experiment_id === 'residual-connections' ? 'Variant not recorded' : 'Small-model learning'
const runName = (run: LearningRun) => `${variantName(run)} · ${run.current_step}/${run.max_steps} updates · ${run.status} · ${run.run_id.slice(0, 8)}`
function pretty(setting: unknown) {
  if (setting === null || setting === undefined) return 'Not recorded'
  const text = typeof setting === 'string' ? setting : JSON.stringify(setting)
  return text.length > 360 ? text.slice(0, 360) + '…' : text
}
function differenceLabel(field: string) {
  if (field === 'model.assembly_sha256') return 'Graph connections'
  if (field === 'variant') return 'Residual connections'
  return field
}
function differenceValue(field: string, setting: unknown, run: LearningRun) {
  if (field === 'model.assembly_sha256' && typeof setting === 'string') return `${variantName(run)} · signature ${setting.slice(0, 12)}…`
  if (field === 'variant' && setting === 'with-residuals') return 'With residuals'
  if (field === 'variant' && setting === 'without-residuals') return 'Without residuals'
  return pretty(setting)
}
function chartPoints(first: LearningRun, second: LearningRun): ComparisonPoint[] {
  const rows = new Map<number, ComparisonPoint>()
  for (const [prefix, run] of [['a', first], ['b', second]] as const) {
    for (const point of run.metrics || []) {
      if (!finite(point.step)) continue
      const row = rows.get(point.step) || { step: point.step }
      for (const [key, measurement] of [['Train', point.language_loss], ['Validation', point.eval_language_loss ?? point.eval_loss], ['Gradient', point.gradient_norm]] as const) {
        if (finite(measurement)) row[`${prefix}${key}`] = measurement
      }
      rows.set(point.step, row)
    }
  }
  return [...rows.values()].sort((a, b) => a.step - b.step)
}
function OverlayChart({ title, points, gradient = false }: { title: string; points: ComparisonPoint[]; gradient?: boolean }) {
  const lines = gradient
    ? [{ key: 'aGradient', label: 'A · pre-clipping gradient norm', color: '#38bdf8' }, { key: 'bGradient', label: 'B · pre-clipping gradient norm', color: '#fb923c' }]
    : [{ key: 'aTrain', label: 'A · training CE', color: '#38bdf8' }, { key: 'aValidation', label: 'A · validation CE', color: '#38bdf8', dash: '5 4' }, { key: 'bTrain', label: 'B · training CE', color: '#fb923c' }, { key: 'bValidation', label: 'B · validation CE', color: '#fb923c', dash: '5 4' }]
  return <section aria-label={title}><h4 className="mb-3 text-sm font-medium text-slate-200">{title}</h4><ResponsiveContainer width="100%" height={240}><LineChart data={points} margin={{ top: 5, right: 12, bottom: 0, left: 0 }}><CartesianGrid stroke="#1e293b" strokeDasharray="3 3" /><XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} tick={{ fill: '#64748b', fontSize: 11 }} /><YAxis domain={[0, 'auto']} tick={{ fill: '#64748b', fontSize: 11 }} width={55} /><Tooltip labelFormatter={step => `Optimizer update ${step}`} contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }} /><Legend wrapperStyle={{ fontSize: 11 }} />{lines.filter(line => points.some(point => finite(point[line.key as keyof ComparisonPoint]))).map(line => <Line key={line.key} dataKey={line.key} name={line.label} stroke={line.color} strokeDasharray={'dash' in line ? line.dash : undefined} dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />)}</LineChart></ResponsiveContainer></section>
}

export default function LearningRunComparison({ runs, runA, runB, onRunAChange, onRunBChange }: Props) {
  const [comparison, setComparison] = useState<LearningComparison | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    setComparison(null); setError('')
    if (!runA || !runB || runA === runB) { setLoading(false); return }
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    setLoading(true)
    async function read() {
      try {
        const result = await compareLearningRuns(runA, runB)
        if (!current) return
        setComparison(result); setLoading(false)
        if (active(result.run_a) || active(result.run_b)) timer = setTimeout(read, 1200)
      } catch (err) {
        if (current) { setError(errMessage(err)); setLoading(false) }
      }
    }
    void read()
    return () => { current = false; clearTimeout(timer) }
  }, [runA, runB, retry])
  const points = comparison ? chartPoints(comparison.run_a, comparison.run_b) : []
  const hasCE = points.some(point => [point.aTrain, point.aValidation, point.bTrain, point.bValidation].some(finite))
  const hasGradient = points.some(point => [point.aGradient, point.bGradient].some(finite))
  const firstSamples = comparison?.run_a.after?.samples || comparison?.run_a.results?.after.samples || []
  const secondSamples = comparison?.run_b.after?.samples || comparison?.run_b.results?.after.samples || []
  const prompts = [...new Set([...firstSamples, ...secondSamples].map(sample => sample.prompt))]

  return <section aria-labelledby="learning-ab" className="space-y-4">
    <div><p className="text-xs font-medium uppercase tracking-widest text-slate-500">A/B · Check the effect of a change</p><h2 id="learning-ab" className="mt-2 text-xl font-semibold">Compare recorded runs</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">Choose two saved runs. Check their settings before interpreting the curves: different data, initial weights or hardware can change the outcome too.</p></div>
    <div className="grid gap-4 sm:grid-cols-2">
      {([{ label: 'Run A', selected: runA, change: onRunAChange }, { label: 'Run B', selected: runB, change: onRunBChange }]).map(control => <label key={control.label} className="min-w-0 space-y-1.5 text-sm text-slate-400"><span>{control.label}</span><select value={control.selected} onChange={event => control.change(event.target.value)} className="w-full min-w-0 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 text-slate-200"><option value="">Choose a run</option>{control.selected && !runs.some(run => run.run_id === control.selected) && <option value={control.selected}>{control.selected.slice(0, 12)} · selected from link</option>}{runs.map(run => <option key={run.run_id} value={run.run_id}>{runName(run)}</option>)}</select></label>)}
    </div>
    {runA && runA === runB && <p role="status" className="text-sm text-amber-200">Choose two different runs for an A/B comparison.</p>}
    {loading && <p role="status" className="text-sm text-slate-400">Loading the two recorded runs…</p>}
    {error && <div role="alert" className="rounded-xl border border-red-500/30 bg-red-950/30 p-4 text-sm text-red-200">{error}<button className="ml-3 underline" onClick={() => setRetry(value => value + 1)}>Retry comparison</button></div>}
    {comparison && <>
      <div className={`rounded-xl border p-4 text-sm ${comparison.comparability.comparable ? 'border-emerald-500/30 bg-emerald-950/20 text-emerald-200' : 'border-amber-500/30 bg-amber-950/20 text-amber-200'}`}>
        <p className="font-medium">{comparison.comparability.comparable ? 'The recorded comparison checks match.' : 'Some comparison conditions differ or were not recorded.'}</p>
        {comparison.comparability.warnings.map((warning, index) => <p key={index} className="mt-2 leading-6">{warning}</p>)}
        <p className="mt-2 leading-6">Small models can learn with either variant. A larger gradient norm or a single lower loss does not by itself establish that an architecture is better.</p>
      </div>
      <details className="rounded-xl border border-slate-800 bg-slate-900/50 p-4"><summary className="cursor-pointer text-sm text-slate-300">Recorded comparison checks · {comparison.comparability.checks.filter(check => check.match === true).length} match · {comparison.comparability.checks.filter(check => check.match === false).length} different · {comparison.comparability.checks.filter(check => check.match === null).length} not recorded</summary><div className="mt-4 overflow-x-auto"><table className="w-full min-w-[520px] text-left text-sm"><caption className="sr-only">Recorded comparison checks</caption><thead className="bg-slate-900/60 text-xs text-slate-500"><tr><th scope="col" className="px-4 py-2">Condition</th><th scope="col" className="px-4 py-2">Result</th><th scope="col" className="px-4 py-2">What was checked</th></tr></thead><tbody>{comparison.comparability.checks.map(check => <tr key={check.id} className="border-t border-slate-800"><th scope="row" className="px-4 py-3 font-normal text-slate-300">{check.label}</th><td className={`px-4 py-3 ${check.match === true ? 'text-emerald-300' : 'text-amber-200'}`}>{check.match === true ? 'Match' : check.match === false ? 'Different' : 'Not recorded'}</td><td className="px-4 py-3 text-xs leading-5 text-slate-400">{check.detail}</td></tr>)}</tbody></table></div></details>
      <details className="rounded-xl border border-slate-800 bg-slate-900/50 p-4"><summary className="cursor-pointer text-sm text-slate-300">Configuration differences ({comparison.configuration_differences.length})</summary>{comparison.configuration_differences.length ? <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[520px] text-left text-xs"><thead className="text-slate-500"><tr><th className="p-2" scope="col">Setting</th><th className="p-2" scope="col">A</th><th className="p-2" scope="col">B</th></tr></thead><tbody>{comparison.configuration_differences.map(row => <tr key={row.field} className="border-t border-slate-800"><th className="p-2 font-normal text-slate-400" scope="row">{differenceLabel(row.field)}</th><td className="max-w-xs break-words p-2 text-slate-300">{differenceValue(row.field, row.a, comparison.run_a)}</td><td className="max-w-xs break-words p-2 text-slate-300">{differenceValue(row.field, row.b, comparison.run_b)}</td></tr>)}</tbody></table></div> : <p className="mt-3 text-xs text-slate-500">No configuration differences were recorded.</p>}</details>
      <div className="grid gap-4 md:grid-cols-2">{([{ label: 'A', run: comparison.run_a, ce: comparison.metrics.validation_cross_entropy.a, ppl: comparison.metrics.validation_perplexity.a }, { label: 'B', run: comparison.run_b, ce: comparison.metrics.validation_cross_entropy.b, ppl: comparison.metrics.validation_perplexity.b }]).map(item => <article key={item.label} className="rounded-xl border border-slate-800 bg-slate-900/60 p-5"><h3 className="font-medium text-slate-100">{item.label} · {variantName(item.run)}</h3><p className="mt-1 text-xs text-slate-500">{item.run.current_step}/{item.run.max_steps} updates · {item.run.status} · {item.run.run_id.slice(0, 12)}</p><dl className="mt-4 grid grid-cols-2 gap-3"><div><dt className="text-xs text-slate-500">Validation cross-entropy</dt><dd className="mt-1 text-lg text-slate-100">{value(item.ce)}</dd></div><div><dt className="text-xs text-slate-500">Validation perplexity</dt><dd className="mt-1 text-lg text-slate-100">{value(item.ppl)}</dd></div></dl></article>)}</div>
      <div className="space-y-6 rounded-xl border border-slate-800 bg-slate-900/60 p-5">{hasCE ? <><OverlayChart title="Training and validation prediction error" points={points} /><p className="text-xs leading-5 text-slate-500">The horizontal axis counts optimizer updates. Solid lines show training cross-entropy; dashed lines show held-out validation cross-entropy. Lower is better only on the same tokenizer and evaluation data.</p></> : <p className="text-sm text-slate-400">No language cross-entropy measurements were recorded for this pair.</p>}{hasGradient ? <><OverlayChart title="Gradient norm before clipping" points={points} gradient /><p className="text-xs leading-5 text-slate-500">Global L2 norm of the parameter gradients, measured before clipping. This is not the clipped norm or a quality score. Missing observations are left absent.</p>{(!points.some(point => finite(point.aGradient)) || !points.some(point => finite(point.bGradient))) && <p className="text-xs text-amber-200">Gradient norms were not recorded for one of these runs.</p>}</> : <p className="text-sm text-slate-400">Gradient norms were not recorded for these runs.</p>}</div>
      {prompts.length > 0 && <div className="space-y-4"><h3 className="font-medium text-slate-200">Generated continuations after each run</h3>{prompts.map(prompt => <article key={prompt} className="overflow-hidden rounded-xl border border-slate-800"><div className="border-b border-slate-800 bg-slate-900 px-4 py-3"><p className="text-xs text-slate-500">Prompt</p><p className="mt-1 text-sm text-slate-200">{prompt}</p></div><div className="grid md:grid-cols-2">{([{ label: 'A', sample: firstSamples.find(sample => sample.prompt === prompt) }, { label: 'B', sample: secondSamples.find(sample => sample.prompt === prompt) }]).map(item => <div key={item.label} className="border-t border-slate-800 p-4 first:border-0 md:border-l md:border-t-0"><p className="mb-2 text-xs font-medium text-slate-500">{item.label}</p><p className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-300">{item.sample ? item.sample.continuation || '(No continuation generated.)' : 'Not recorded for this prompt'}</p></div>)}</div></article>)}</div>}
    </>}
  </section>
}
