import { useEffect, useRef, useState } from 'react'
import { ArrowLeft, ArrowRight, BookOpen, CheckCircle2, ExternalLink, FlaskConical, Play, Square, Upload } from 'lucide-react'
import MetricsPanel from './MetricsPanel'
import LearningRunComparison from './LearningRunComparison'
import ResidualDiagram from './ResidualDiagram'
import { errMessage } from '../services/api'
import {
  getLearningExperiment, getLearningRun, listLearningRuns, parseLearningPreset, startLearningRun, stopLearningRun,
  LEARNING_EXPERIMENT_ID, RESIDUAL_EXPERIMENT_ID, type LearningDescriptor, type LearningDevice, type LearningObservation, type LearningRun, type LearningExperimentId, type LearningPreset, type ResidualVariant,
} from '../services/learning'

const ACTIVE = new Set(['starting', 'preparing', 'training', 'stopping', 'evaluating', 'sampling'])
const CONCEPTS = [
  { id: 'bpe', title: 'Turn text into tokens', explanation: 'A tokenizer turns the supplied sentences into the units the model predicts.' },
  { id: 'transformer', title: 'Follow the computation', explanation: 'Two small decoder blocks use preceding tokens to produce next-token scores.' },
  { id: 'adam', title: 'Update the weights', explanation: 'AdamW adjusts the parameters using prediction errors on training examples.' },
  { id: 'gpt1', title: 'Understand pretraining', explanation: 'Predicting the next token provides the learning objective, without instruction answers.' },
]
const PANEL = 'rounded-2xl border border-slate-800 bg-slate-900/70'
const number = (value: number | null | undefined, digits = 3) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('en-US', { maximumFractionDigits: digits }) : 'Not measured yet'
const validationCE = (value?: LearningObservation | null) => value?.metrics.val_language_loss ?? value?.metrics.val_loss
const initialRun = () => new URLSearchParams(window.location.hash.slice(1)).get('run') || ''

function saveSelection(experimentId: LearningExperimentId, runId: string, compareId: string) {
  const hash = new URLSearchParams({ experiment: experimentId })
  if (runId) hash.set('run', runId)
  if (compareId) hash.set('compare', compareId)
  window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}#${hash}`)
}
function phaseLabel(run: LearningRun | null) {
  if (!run) return 'Choose a run or start a new experiment'
  const phase = run.phase || run.status
  return phase.replace(/[_-]/g, ' ').replace(/^\w/, c => c.toUpperCase())
}
function datedRun(run: LearningRun) {
  const date = new Date(run.created_at)
  return `${Number.isNaN(date.getTime()) ? run.run_id.slice(0, 8) : date.toLocaleString()} · ${run.status} · ${run.current_step}/${run.max_steps} updates`
}
function readFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(new Error('Could not read the recipe file.'))
    reader.readAsText(file, 'UTF-8')
  })
}

interface Props {
  onBack: () => void
  experimentId?: LearningExperimentId
  initialPreset?: LearningPreset
  onOpenExperiment?: (id: LearningExperimentId, preset?: LearningPreset) => void
}
export default function LearningExperiment({ onBack, experimentId = LEARNING_EXPERIMENT_ID, initialPreset, onOpenExperiment }: Props) {
  const residual = experimentId === RESIDUAL_EXPERIMENT_ID
  const [descriptor, setDescriptor] = useState<LearningDescriptor | null>(null)
  const [runs, setRuns] = useState<LearningRun[]>([])
  const [selected, setSelected] = useState(initialRun)
  const [compareId, setCompareId] = useState(() => new URLSearchParams(window.location.hash.slice(1)).get('compare') || '')
  const [run, setRun] = useState<LearningRun | null>(null)
  const [steps, setSteps] = useState(120)
  const [seed, setSeed] = useState(42)
  const [device, setDevice] = useState<LearningDevice>('cpu')
  const [variant, setVariant] = useState<ResidualVariant>('with-residuals')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [runError, setRunError] = useState('')
  const [notice, setNotice] = useState('')
  const [retry, setRetry] = useState(0)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    let current = true
    getLearningExperiment(experimentId).then(value => {
      if (!current) return
      setDescriptor(value); setSteps(initialPreset?.steps ?? value.default_steps); setSeed(initialPreset?.seed ?? value.seed)
      setDevice(value.default_device || value.device)
      setVariant(initialPreset?.variant ?? value.default_variant ?? 'with-residuals')
      if (initialPreset) setNotice('Recipe loaded. Review the settings and choose Start experiment when you are ready.')
    }).catch(err => { if (current) setError(`Could not load this experience: ${errMessage(err)}`) })
    listLearningRuns(experimentId).then(value => { if (current) setRuns(value) })
      .catch(err => { if (current) setError(`Could not load recorded runs: ${errMessage(err)}`) })
    return () => { current = false; mounted.current = false }
  }, [experimentId, initialPreset])

  useEffect(() => {
    const change = () => { setSelected(initialRun()); setCompareId(new URLSearchParams(window.location.hash.slice(1)).get('compare') || '') }
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])

  useEffect(() => {
    if (!selected) { setRun(null); setRunError(''); return }
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    setLoading(true); setRun(null); setRunError('')
    async function refresh() {
      try {
        const value = await getLearningRun(selected)
        if (!current) return
        setRun(value); setLoading(false); setRunError('')
        setRuns(previous => [value, ...previous.filter(item => item.run_id !== value.run_id)])
        if (ACTIVE.has(value.status)) timer = setTimeout(refresh, 1200)
      } catch (err) {
        if (current) { setLoading(false); setRunError(`Could not read this run: ${errMessage(err)}`) }
      }
    }
    void refresh()
    return () => { current = false; clearTimeout(timer) }
  }, [selected, retry])

  function selectRun(id: string) {
    setSelected(id); saveSelection(experimentId, id, compareId); setError(''); setNotice('')
  }
  function selectComparison(id: string) { setCompareId(id); saveSelection(experimentId, selected, id) }
  const active = Boolean(run && ACTIVE.has(run.status))
  const otherActive = runs.some(item => item.run_id !== selected && ACTIVE.has(item.status))
  const available = descriptor?.available_devices || (descriptor ? [descriptor.device] : [])
  const valid = Number.isInteger(steps) && steps >= (descriptor?.min_steps ?? 20) && steps <= (descriptor?.max_steps ?? 300)
    && Number.isInteger(seed) && seed >= 0 && seed <= 2147483647

  async function start() {
    if (!descriptor || !valid || busy || active || otherActive) return
    setBusy(true); setError(''); setNotice('')
    try {
      const value = await startLearningRun({ steps, seed, device, ...(residual ? { variant } : {}) }, experimentId)
      if (!mounted.current) return
      selectRun(value.run_id)
      setRetry(value => value + 1)
    } catch (err) { if (mounted.current) setError(errMessage(err)) }
    finally { if (mounted.current) setBusy(false) }
  }
  async function stop() {
    if (!run || busy || run.status === 'stopping') return
    setBusy(true); setError('')
    try {
      await stopLearningRun(run.run_id)
      if (!mounted.current) return
      setRun(value => value ? { ...value, status: 'stopping', phase: 'stopping' } : value)
      setRetry(value => value + 1)
    } catch (err) { if (mounted.current) setError(errMessage(err)) }
    finally { if (mounted.current) setBusy(false) }
  }
  async function importRecipe(file?: File) {
    if (!file) return
    setError(''); setNotice('')
    if (file.size > 65536) { setError('The recipe file must be 64 KB or smaller.'); return }
    try {
      const preset = parseLearningPreset(await readFile(file))
      if (!mounted.current) return
      if (preset.experiment_id !== experimentId) {
        if (!onOpenExperiment) throw new Error('Open the matching guided experiment before importing this recipe.')
        onOpenExperiment(preset.experiment_id, preset)
        return
      }
      setSteps(preset.steps); setSeed(preset.seed)
      if (residual) setVariant(preset.variant ?? 'with-residuals')
      setNotice('Recipe loaded. Review the settings and choose Start experiment when you are ready.')
    } catch (err) { if (mounted.current) setError(errMessage(err)) }
  }

  const before = run?.before || run?.results?.before
  const after = run?.after || run?.results?.after
  const prompts = descriptor?.prompts || before?.samples.map(sample => sample.prompt) || []
  const percent = run && run.max_steps > 0 ? Math.min(100, Math.max(0, 100 * run.current_step / run.max_steps)) : 0
  const hardware = run?.configuration?.hardware
  const batch = run?.configuration?.training?.batch_size
  const accumulation = run?.configuration?.training?.gradient_accumulation_steps
  const sequence = run?.configuration?.model?.max_seq_len
  const tokenBudget = run && batch && accumulation && sequence ? run.max_steps * batch * accumulation * (sequence - 1) : undefined
  const elapsed = run?.elapsed_seconds
  const throughput = typeof elapsed === 'number' && elapsed > 0 && typeof run?.tokens_seen === 'number'
    ? run.tokens_seen / elapsed : undefined
  const concepts = residual ? [
    { id: 'transformer', title: 'Follow the bypass paths', explanation: 'Residual additions combine a transformed representation with the representation that entered the branch.' },
    { id: 'pre-layernorm', title: 'Keep normalization before the branch', explanation: 'Both variants use the same pre-norm placement. This experiment changes the bypass additions.' },
    { id: 'rmsnorm', title: 'Read the normalization', explanation: 'RMSNorm remains inside each attention and feed-forward branch.' },
    { id: 'adam', title: 'Observe optimizer updates', explanation: 'The optimizer updates the weights using gradients. The measured norm is taken before clipping.' },
  ] : CONCEPTS

  function prepareOtherVariant() {
    const saved = run?.configuration
    if (!saved || active || busy) return
    if (saved.steps != null) setSteps(saved.steps)
    if (saved.seed != null) setSeed(saved.seed)
    setVariant(saved.variant === 'without-residuals' ? 'with-residuals' : 'without-residuals')
    if (saved.device && available.includes(saved.device)) setDevice(saved.device)
    setNotice(saved.device && !available.includes(saved.device)
      ? 'The other variant is prepared with the recorded budget and seed. Its recorded device is unavailable; choose an available device explicitly before starting.'
      : 'The other variant is prepared with the recorded budget, seed and device. Review the settings, then start it explicitly.')
  }

  return <div className="min-h-screen bg-slate-950 text-slate-200">
    <header className="border-b border-slate-800 bg-slate-900/60">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-6">
        <button onClick={onBack} className="flex items-center gap-2 rounded-lg px-2 py-1 text-sm text-slate-400 hover:text-white"><ArrowLeft size={16} />Projects</button>
        <span className="flex items-center gap-2 text-sm font-medium"><FlaskConical size={18} className="text-accent" />LLM Lab · Learning experience {residual ? '02' : '01'}</span>
        <a href={`https://www.thelastserver.com/understanding-ai/#tech-${residual ? 'transformer' : 'gpt1'}`} target="_blank" rel="noreferrer" className="flex items-center gap-2 text-sm text-accent hover:underline"><BookOpen size={15} />{residual ? 'How residual connections work' : 'How pretraining works'}<ExternalLink size={12} /></a>
      </div>
    </header>
    <main className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6">
      <section className="max-w-3xl">
        <p className="mb-3 text-xs font-medium uppercase tracking-[.18em] text-accent">From an idea to a measured experiment</p>
        <h1 className="text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">{residual ? 'What do residual connections change?' : 'What does a small model learn?'}</h1>
        <p className="mt-4 text-base leading-7 text-slate-400">{residual ? 'Train one version with bypass additions around attention and the feed-forward network, then train the version without those additions. Compare prediction errors and pre-clipping gradient norms, checking the saved conditions before drawing a conclusion.' : 'Start with random weights, train on a small supplied corpus, and compare predictions before and after. Watch the error on separate validation sentences to see whether learning transfers beyond the training examples.'}</p>
      </section>

      {error && <div role="alert" className="rounded-xl border border-red-500/30 bg-red-950/30 p-4 text-sm text-red-200">{error}</div>}
      {notice && <p role="status" className="rounded-xl border border-accent/30 bg-accent/5 p-4 text-sm text-accent">{notice}</p>}

      <section aria-labelledby="learning-setup" className={`${PANEL} p-5 sm:p-7`}>
        <div className="grid gap-7 lg:grid-cols-[1.15fr_1fr]">
          <div>
            <p className="text-xs font-medium uppercase tracking-widest text-slate-500">01 · Prepare</p>
            <h2 id="learning-setup" className="mt-2 text-xl font-semibold">A small, controlled learning task</h2>
            <p className="mt-3 text-sm leading-6 text-slate-400">{descriptor?.dataset.description || 'The supplied artificial corpus contains short sentences with recurring patterns. Training and validation use distinct documents. The tokenizer learns from training text only.'}</p>
            <div className="mt-4 flex flex-wrap gap-2 text-xs">
              <span className="rounded-full border border-slate-700 px-3 py-1.5">{descriptor ? `${number(descriptor.dataset.train_documents, 0)} training documents` : 'Loading corpus details…'}</span>
              {descriptor && <span className="rounded-full border border-slate-700 px-3 py-1.5">{number(descriptor.dataset.validation_documents, 0)} validation documents</span>}
              <span className="rounded-full border border-slate-700 px-3 py-1.5">No model download</span>
            </div>
            <p className="mt-4 text-xs leading-5 text-slate-500">This exercise measures learning on a toy corpus. Lower validation error here does not establish general language understanding or factual knowledge.</p>
            {descriptor && <p className="mt-3 text-xs text-slate-500">Model: {descriptor.model.n_layers} decoder blocks · width {descriptor.model.d_model} · context {descriptor.model.max_seq_len} tokens · locally fitted byte-level BPE.</p>}
          </div>
          <div className="space-y-4">
            {residual && <div className="space-y-1.5 text-sm text-slate-400">
              <label htmlFor="learning-residual-variant">Residual connections</label>
              <select id="learning-residual-variant" aria-describedby="learning-residual-help" value={variant} disabled={!descriptor || busy || active} onChange={event => setVariant(event.target.value as ResidualVariant)} className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 disabled:opacity-50">{(descriptor?.variants || [{ id: 'with-residuals', title: 'With residuals', description: '' }, { id: 'without-residuals', title: 'Without residuals', description: '' }]).map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select>
              <p id="learning-residual-help" className="text-xs leading-5 text-slate-500">Run the first variant, wait for it to finish, then start the other variant explicitly. Select the two recorded runs under A/B comparison.</p>
            </div>}
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="space-y-1.5 text-sm text-slate-400"><span>Optimizer updates</span>
                <input type="number" min={descriptor?.min_steps ?? 20} max={descriptor?.max_steps ?? 300} step="1" value={Number.isNaN(steps) ? '' : steps} onChange={event => setSteps(event.target.value === '' ? NaN : Number(event.target.value))} disabled={!descriptor || busy || active} className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 disabled:opacity-50" />
              </label>
              <label className="space-y-1.5 text-sm text-slate-400"><span>Random seed</span>
                <input type="number" min="0" max="2147483647" step="1" value={Number.isNaN(seed) ? '' : seed} onChange={event => setSeed(event.target.value === '' ? NaN : Number(event.target.value))} disabled={!descriptor || busy || active} className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 disabled:opacity-50" />
              </label>
            </div>
            <label className="block space-y-1.5 text-sm text-slate-400"><span>Compute device</span>
              <select value={device} onChange={event => setDevice(event.target.value as LearningDevice)} disabled={!descriptor || busy || active} className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-slate-100 disabled:opacity-50">
                {available.length === 0 && <option value="cpu">Loading available hardware…</option>}
                {available.map(value => <option key={value} value={value}>{value === 'cuda' ? `NVIDIA GPU${descriptor?.hardware?.cuda_device_name ? ` · ${descriptor.hardware.cuda_device_name}` : ''}` : `CPU${descriptor?.hardware?.cpu_name ? ` · ${descriptor.hardware.cpu_name}` : ''}`}</option>)}
              </select>
            </label>
            <p className="text-xs leading-5 text-slate-500">One update adjusts the weights using a batch of training examples. The selected update budget is not a duration estimate. Keep the seed and settings fixed when comparing runs.</p>
            {!valid && <p className="text-xs text-amber-300">Choose {descriptor?.min_steps ?? 20}–{descriptor?.max_steps ?? 300} updates and an integer seed from 0 to 2147483647.</p>}
            <div className="flex flex-wrap items-center gap-3">
              <button onClick={() => void start()} disabled={!descriptor || !valid || busy || active || otherActive || loading} className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-accent-dim disabled:cursor-not-allowed disabled:opacity-40"><Play size={15} />{busy && !active ? 'Starting…' : 'Start experiment'}</button>
              <label className={`flex cursor-pointer items-center gap-2 rounded-lg border border-slate-700 px-3 py-2.5 text-sm text-slate-300 ${!descriptor || active || busy ? 'pointer-events-none opacity-40' : 'hover:border-slate-500'}`}><Upload size={15} />Import recipe
                <input type="file" accept="application/json,.json" aria-label="Import experiment recipe" className="sr-only" disabled={!descriptor || active || busy} onChange={event => { void importRecipe(event.target.files?.[0]); event.target.value = '' }} />
              </label>
            </div>
            <p className="text-xs leading-5 text-slate-500">Starting creates a new recorded run. Previous runs remain available below.</p>
            {residual && run?.configuration?.variant && <button type="button" disabled={busy || active} onClick={prepareOtherVariant} className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:border-slate-500 disabled:opacity-40">Prepare the other variant from this run</button>}
            {otherActive && <p role="status" className="text-xs text-amber-200">Another recorded learning run is active. Select it below to follow or stop it.</p>}
          </div>
        </div>
        {residual && <div className="mx-auto mt-6 max-w-4xl"><ResidualDiagram variant={variant} layers={descriptor?.model.n_layers} /><p className="mt-3 text-xs leading-5 text-slate-500">A small model can learn with either arrangement. This experiment observes how the change behaves under a short training budget; it does not guarantee that removing the bypasses will cause failure.</p></div>}
      </section>

      <section aria-labelledby="learning-observe" className="space-y-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div><p className="text-xs font-medium uppercase tracking-widest text-slate-500">02 · Observe</p><h2 id="learning-observe" className="mt-2 text-xl font-semibold">Follow the learning, not just the progress bar</h2></div>
          <label className="min-w-0 text-sm text-slate-400"><span className="mb-1.5 block">Recorded experiment</span>
            <select value={selected} onChange={event => selectRun(event.target.value)} className="max-w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-200 sm:max-w-md">
              <option value="">Choose a recorded run</option>
              {selected && !runs.some(item => item.run_id === selected) && <option value={selected}>Loading selected run…</option>}
              {runs.map(item => <option key={item.run_id} value={item.run_id}>{datedRun(item)}</option>)}
            </select>
          </label>
        </div>
        {runError && <div role="alert" className="flex flex-wrap items-center gap-3 rounded-xl border border-red-500/30 bg-red-950/30 p-4 text-sm text-red-200">{runError}<button onClick={() => setRetry(value => value + 1)} className="underline">Retry loading run</button></div>}
        <div className={`${PANEL} p-5 sm:p-6`}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div><p role="status" className="font-medium text-slate-100">{loading ? 'Loading saved observations…' : phaseLabel(run)}</p>
              {run && <p className="mt-1 text-xs text-slate-500">Run {run.run_id.slice(0, 12)} · {run.current_step} / {run.max_steps} optimizer updates · seed {run.configuration?.seed ?? 'not recorded'}{run.configuration?.variant ? ` · ${run.configuration.variant === 'with-residuals' ? 'with residuals' : 'without residuals'}` : ''}</p>}
            </div>
            {active && <button onClick={() => void stop()} disabled={busy || run?.status === 'stopping'} className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200 disabled:opacity-40"><Square size={14} />{run?.status === 'stopping' ? 'Stopping safely…' : 'Stop experiment'}</button>}
            {run?.status === 'completed' && <span className="flex items-center gap-1.5 text-sm text-accent"><CheckCircle2 size={16} />Run completed</span>}
          </div>
          {run && <><progress aria-label="Training updates completed" max={run.max_steps || 1} value={run.current_step} className="mt-4 h-2 w-full accent-emerald-400" /><p className="mt-1 text-right text-xs text-slate-500">{number(percent, 0)}%</p></>}
          {run?.error && <p role="alert" className="mt-4 rounded-lg bg-red-950/40 p-3 text-sm text-red-200">{run.error}</p>}
          {run && ['stopped', 'interrupted', 'error', 'failed'].includes(run.status) && <p className="mt-3 text-sm text-amber-200">This run did not complete its full budget. The saved observations below describe only the work that actually finished.</p>}
          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: 'Parameters', value: number(before?.parameters, 0), detail: 'Counted on the actual model.' },
              { label: 'Training tokens seen', value: number(run?.tokens_seen, 0), detail: `${run?.tokens_measured_step != null ? `Last recorded at update ${run.tokens_measured_step}. ` : ''}${tokenBudget ? `Planned budget: approximately ${number(tokenBudget, 0)} tokens.` : 'Reported by the training process.'}` },
              { label: 'Elapsed time', value: typeof elapsed === 'number' && Number.isFinite(elapsed) ? `${number(elapsed, 1)} s` : 'Not measured yet', detail: 'Measured for this run; hardware affects duration.' },
              { label: 'Average throughput', value: throughput === undefined ? 'Not measured yet' : `${number(throughput, 1)} tokens/s`, detail: 'Last recorded training tokens divided by elapsed run time; includes non-training work.' },
            ].map(item => <div key={item.label} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4"><p className="text-xs text-slate-400">{item.label}</p><p className="mt-2 text-xl font-semibold text-slate-100">{item.value}</p><p className="mt-2 text-xs leading-5 text-slate-500">{item.detail}</p></div>)}
          </div>
          {hardware && <p className="mt-4 text-xs leading-5 text-slate-500">Measured on {hardware.device_name || hardware.device || run?.configuration?.device}; {hardware.dtype || 'precision not recorded'}{hardware.threads ? `; ${hardware.threads} CPU threads` : ''}{hardware.torch_version ? `; PyTorch ${hardware.torch_version}` : ''}.</p>}
          {run && <details className="mt-5 border-t border-slate-800 pt-4"><summary className="cursor-pointer text-sm text-slate-400">Saved run log ({run.logs?.length ?? 0} entries)</summary><div className="mt-3 max-h-64 overflow-auto rounded-lg bg-slate-950 p-3 font-mono text-xs leading-6 text-slate-400">{run.logs?.length ? run.logs.map((entry, i) => <p key={i} className={entry.level === 'error' ? 'text-red-300' : ''}>{entry.message}</p>) : <p>No log messages saved yet.</p>}</div></details>}
        </div>
        {run && run.metrics?.length > 0 && <div className={`${PANEL} p-5 sm:p-6`}><MetricsPanel metrics={run.metrics} metricDefinition="language_cross_entropy_v2" simpleLanguageView /></div>}
      </section>

      <section aria-labelledby="learning-compare" className="space-y-4">
        <div><p className="text-xs font-medium uppercase tracking-widest text-slate-500">03 · Compare</p><h2 id="learning-compare" className="mt-2 text-xl font-semibold">The same questions, before and after</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">The prompts and generation settings stay fixed. Read the continuations together with validation error: a fluent-looking sentence alone does not establish that the model generalizes.</p></div>
        <div className="grid gap-4 md:grid-cols-2">
          {[{ title: 'Before training', data: before }, { title: after ? `After ${run?.results?.step ?? run?.current_step} updates` : 'After training', data: after }].map(item => <div key={item.title} className={`${PANEL} p-5`}><h3 className="font-semibold text-slate-100">{item.title}</h3><dl className="mt-4 grid grid-cols-2 gap-3"><div><dt className="text-xs text-slate-500">Validation cross-entropy</dt><dd className="mt-1 text-lg font-medium">{number(validationCE(item.data))}</dd></div><div><dt className="text-xs text-slate-500">Validation perplexity</dt><dd className="mt-1 text-lg font-medium">{number(item.data?.metrics.val_perplexity)}</dd></div></dl></div>)}
        </div>
        <p className="text-xs leading-5 text-slate-500">Cross-entropy is the average prediction error per validation token, in nats; lower is better on the same evaluation data and tokenizer. Perplexity is its exponential. Validation sentences are held out from training.</p>
        {prompts.map(prompt => {
          const first = before?.samples.find(sample => sample.prompt === prompt)
          const last = after?.samples.find(sample => sample.prompt === prompt)
          return <article key={prompt} className={`${PANEL} overflow-hidden`}><div className="border-b border-slate-800 px-5 py-3"><h3 className="text-xs font-medium uppercase tracking-wider text-slate-500">Fixed prompt</h3><p className="mt-1 whitespace-pre-wrap text-sm text-slate-200">{prompt}</p></div><div className="grid md:grid-cols-2"><div className="p-5"><p className="mb-3 text-xs font-medium text-slate-500">Before</p><p className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-300">{first ? first.continuation || '(No continuation generated.)' : 'Not measured yet'}</p></div><div className="border-t border-slate-800 bg-accent/[.025] p-5 md:border-l md:border-t-0"><p className="mb-3 text-xs font-medium text-accent">After</p><p className="whitespace-pre-wrap break-words text-sm leading-7 text-slate-300">{last ? last.continuation || '(No continuation generated.)' : 'Not measured yet'}</p></div></div></article>
        })}
      </section>

      <LearningRunComparison runs={runs} runA={selected} runB={compareId} onRunAChange={selectRun} onRunBChange={selectComparison} />

      <section aria-labelledby="learning-concepts" className="space-y-4"><h2 id="learning-concepts" className="text-xl font-semibold">Connect each step to the idea</h2><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{concepts.map(concept => <a key={concept.id} href={`https://www.thelastserver.com/understanding-ai/#tech-${concept.id}`} target="_blank" rel="noreferrer" className={`${PANEL} p-5 transition-colors hover:border-accent/50`}><h3 className="flex items-center justify-between gap-2 text-sm font-semibold text-accent">{concept.title}<ArrowRight size={14} /></h3><p className="mt-3 text-xs leading-6 text-slate-400">{concept.explanation}</p></a>)}</div></section>
    </main>
  </div>
}
