import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'
import type { MetricPoint } from '../types'

interface Props { metrics: MetricPoint[]; metricDefinition?: string | null; simpleLanguageView?: boolean }
const tooltipStyle = { backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0', fontSize: 12 }
const tick = { fontSize: 10, fill: '#64748b' }
const format = (value: number) => Math.abs(value) >= 10000 || (value !== 0 && Math.abs(value) < .001) ? value.toExponential(2) : value.toLocaleString('en-US', { maximumFractionDigits: 3 })

function Chart({ title, data, series, logarithmic = false }: {
  title: string; data: MetricPoint[]; series: { key: keyof MetricPoint; name: string; color: string }[]; logarithmic?: boolean
}) {
  return <section aria-label={title}>
    <h4 className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-400">{title}</h4>
    <ResponsiveContainer width="100%" height={190}>
      <LineChart data={data} margin={{ top: 6, right: 12, left: 4, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="step" type="number" domain={['dataMin', 'dataMax']} tick={tick} tickLine={false} axisLine={{ stroke: '#334155' }} />
        <YAxis scale={logarithmic ? 'log' : 'auto'} domain={logarithmic ? ['auto', 'auto'] : [0, 'auto']} tick={tick} tickLine={false} axisLine={false} width={66} tickFormatter={format} />
        <Tooltip contentStyle={tooltipStyle} labelFormatter={v => `Step ${v}`} formatter={(v, name) => [typeof v === 'number' ? format(v) : String(v), name]} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map(s => <Line key={s.key} name={s.name} type="linear" dataKey={s.key} stroke={s.color} strokeWidth={2} dot={{ r: 2 }} activeDot={{ r: 4 }} connectNulls isAnimationActive={false} />)}
      </LineChart>
    </ResponsiveContainer>
  </section>
}

export default function MetricsPanel({ metrics, metricDefinition, simpleLanguageView = false }: Props) {
  const languageMetrics = metricDefinition === 'language_cross_entropy_v2'
  const [logarithmic, setLogarithmic] = useState(true)
  const [showMore, setShowMore] = useState(false)
  const lossData = metrics.filter(m => m.loss !== undefined || m.eval_loss !== undefined)
  const pplData = metrics.filter(m => m.perplexity !== undefined || m.eval_perplexity !== undefined)
  const lrData = metrics.filter(m => m.lr !== undefined)
  const speedData = metrics.filter(m => m.tokens_per_second !== undefined)
  const lastTrain = [...metrics].reverse().find(m => m.perplexity !== undefined)
  const lastVal = [...metrics].reverse().find(m => m.eval_perplexity !== undefined)
  if (!metrics.length) return <div className="flex h-48 items-center justify-center text-sm text-slate-500">Waiting for metrics…</div>
  const simple = simpleLanguageView && languageMetrics
  const auxiliary = metrics.some(m => [m.auxiliary_loss, m.eval_auxiliary_loss]
    .some(value => typeof value === 'number' && Number.isFinite(value) && (!simple || value !== 0)))
  const measurements = <>
    <section aria-label="Perplexity monitoring" className="space-y-3">
      <div className="grid grid-cols-2 gap-2 text-xs">
        {[{ title: 'Training perplexity', point: lastTrain, value: lastTrain?.perplexity }, { title: 'Validation perplexity', point: lastVal, value: lastVal?.eval_perplexity }].map(s => <div key={s.title} className="rounded-lg border border-slate-800 bg-slate-900 p-3">
          <div className="text-slate-400">{s.title}</div><div className="mt-1 text-lg font-semibold text-slate-200">{s.value === undefined ? 'Waiting…' : format(s.value)}</div>
          {s.point && <div className="text-slate-500">Step {s.point.step.toLocaleString('en-US')}</div>}
        </div>)}
      </div>
      <p className="text-xs leading-5 text-slate-400">Lower perplexity means the model predicts the next token with less uncertainty. Compare runs using the same tokenizer and evaluation data.</p>
      {languageMetrics ? <p className="text-xs leading-5 text-slate-400">Perplexity is exp(token-weighted cross-entropy). Auxiliary objectives and z-loss are excluded. Missing or overflowing values are not plotted.</p> : <p className="text-xs leading-5 text-amber-200/70">Legacy metrics: reported perplexity may include auxiliary objectives. Treat it as an approximation and do not compare it directly with language perplexity from newer runs.</p>}
      <label className="flex items-center gap-2 text-xs text-slate-400"><input type="checkbox" checked={logarithmic} onChange={e => setLogarithmic(e.target.checked)} />Logarithmic perplexity scale</label>
      {pplData.length > 0 ? <Chart title="Reported perplexity" data={pplData} logarithmic={logarithmic} series={[{ key: 'perplexity', name: 'Training perplexity', color: '#a78bfa' }, { key: 'eval_perplexity', name: 'Validation perplexity', color: '#f59e0b' }]} /> : <p className="text-xs text-slate-500">Waiting for reported perplexity…</p>}
    </section>
    {languageMetrics && auxiliary && <Chart title="Auxiliary objectives" data={metrics} series={[{ key: 'auxiliary_loss', name: 'Training auxiliary loss', color: '#f472b6' }, { key: 'eval_auxiliary_loss', name: 'Validation auxiliary loss', color: '#fb923c' }]} />}
    {lrData.length > 0 && <Chart title="Learning rate" data={lrData} series={[{ key: 'lr', name: 'Learning rate', color: '#3b82f6' }]} />}
    {speedData.length > 0 && <Chart title="Throughput" data={speedData} series={[{ key: 'tokens_per_second', name: 'Tokens / second', color: '#22d3ee' }]} />}
  </>
  if (simple) {
    const predictionData = metrics.map(point => ({ ...point, eval_language_loss: point.eval_language_loss ?? point.eval_loss }))
      .filter(point => point.language_loss !== undefined || point.eval_language_loss !== undefined)
    return <div className="space-y-5">
      {predictionData.length > 0 && <Chart title="Next-token prediction error" data={predictionData} series={[
        { key: 'language_loss', name: 'Training cross-entropy', color: '#38bdf8' },
        { key: 'eval_language_loss', name: 'Validation cross-entropy', color: '#f59e0b' },
      ]} />}
      <p className="text-xs leading-5 text-slate-400">Cross-entropy measures prediction error per token. Compare training and held-out validation: lower training error alone does not show that learning transfers to unseen examples.</p>
      <details className="border-t border-slate-800 pt-4" onToggle={event => setShowMore(event.currentTarget.open)}>
        <summary className="cursor-pointer text-sm text-slate-400">More training measurements</summary>
        {showMore && <div className="mt-5 space-y-6">{measurements}</div>}
      </details>
    </div>
  }
  return <div className="space-y-6">
    <p className="text-xs text-slate-500">Charts replay the saved events for the selected run, including after reopening this view.</p>
    {lossData.length > 0 && <Chart title="Loss" data={lossData} series={[{ key: 'loss', name: languageMetrics ? 'Training objective' : 'Training loss', color: '#22c55e' }, ...(languageMetrics ? [{ key: 'language_loss' as const, name: 'Training cross-entropy', color: '#38bdf8' }] : []), { key: 'eval_loss', name: languageMetrics ? 'Validation cross-entropy' : 'Validation loss', color: '#f59e0b' }]} />}
    {measurements}
  </div>
}
