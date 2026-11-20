import { cloneElement, isValidElement, useId, useCallback, useEffect, useState } from 'react'
import {
  ArrowLeft,
  Check,
  FlaskConical,
  Loader2,
  MessageSquare,
  Rocket,
  Upload,
  GitMerge,
  AlertCircle,
  ExternalLink,
} from 'lucide-react'
import {
  errMessage,
  listRuns,
  listExportCheckpoints,
  detectExportFormat,
  exportLocal,
  exportToHub,
  listMergeAvailable,
  mergeLora,
  listCheckpoints,
  generateText,
  type CheckpointInfo,
  type AdapterInfo,
  type FormatDetection,
  type ExportResult,
} from '../services/api'

import type { TrainingStatus } from '../types'

// ─── Shared primitives (same look as TrainingStep) ────────────────────────────

function Field({ label, help, children }: { label: string; help?: string; children: React.ReactNode }) {
  const id = useId()
  const control = isValidElement(children) ? cloneElement(children as React.ReactElement<{ id?: string }>, { id }) : children
  return (
    <div className="flex items-start gap-3">
      <label htmlFor={id} className="text-sm text-slate-400 w-44 flex-none pt-2">{label}</label>
      <div className="flex-1">
        {control}
        {help && <p className="text-xs text-slate-600 mt-1">{help}</p>}
      </div>
    </div>
  )
}

const inputCls =
  'w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-hidden focus:border-accent text-sm'

function CheckpointSelect({
  id,
  checkpoints,
  value,
  onChange,
}: {
  id?: string
  checkpoints: CheckpointInfo[]
  value: string
  onChange: (path: string) => void
}) {
  return (
    <select id={id} value={value} onChange={(e) => onChange(e.target.value)} className={inputCls + ' appearance-none'}>
      <option value="">Select a checkpoint…</option>
      {checkpoints.map((c) => (
        <option key={c.path} value={c.path}>
          {c.name} {c.type ? `· ${c.type}` : ''} {c.step !== undefined ? `· step ${c.step}` : ''} · {c.size_mb} MB
        </option>
      ))}
    </select>
  )
}

function ResultBanner({ result }: { result: { success: boolean; message: string } | null }) {
  if (!result) return null
  return (
    <div
      className={`flex items-start gap-2 text-sm rounded-lg border px-3 py-2 ${
        result.success
          ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
          : 'bg-red-500/10 border-red-500/30 text-red-300'
      }`}
    >
      {result.success ? <Check size={16} className="mt-0.5 flex-none" /> : <AlertCircle size={16} className="mt-0.5 flex-none" />}
      <span className="break-all">{result.message}</span>
    </div>
  )
}



// ─── Export tab ───────────────────────────────────────────────────────────────

function ExportTab({ runId }: { runId: string }) {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [checkpoint, setCheckpoint] = useState('')
  const [detection, setDetection] = useState<FormatDetection | null>(null)
  const [detecting, setDetecting] = useState(false)

  const [repoId, setRepoId] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [isPrivate, setIsPrivate] = useState(true)
  const [modelName, setModelName] = useState('')
  const [description, setDescription] = useState('')
  const [license, setLicense] = useState('apache-2.0')
  const [outputDir, setOutputDir] = useState('/app/data/export')

  const [busy, setBusy] = useState<'local' | 'hub' | null>(null)
  const [result, setResult] = useState<ExportResult | null>(null)

  useEffect(() => {
    listExportCheckpoints(runId).then(setCheckpoints).catch(() => setCheckpoints([]))
  }, [runId])

  useEffect(() => {
    setDetection(null)
    setResult(null)
    if (!checkpoint) return
    setDetecting(true)
    let stale = false
    detectExportFormat(checkpoint)
      .then((d) => { if (!stale) setDetection(d) })
      .catch(() => { if (!stale) setDetection(null) })
      .finally(() => { if (!stale) setDetecting(false) })
    return () => { stale = true }
  }, [checkpoint])

  const doExport = useCallback(
    async (target: 'local' | 'hub') => {
      setBusy(target)
      setResult(null)
      try {
        const payload = {
          checkpoint_path: checkpoint,
          model_name: modelName || undefined,
          model_description: description || undefined,
          license_type: license,
          ...(target === 'local'
            ? { output_dir: outputDir }
            : { repo_id: repoId, hf_token: hfToken, private: isPrivate }),
        }
        const res = target === 'local' ? await exportLocal(payload) : await exportToHub(payload)
        setResult(res)
      } catch (err: unknown) {
        setResult({ success: false, message: errMessage(err) })
      } finally {
        setBusy(null)
      }
    },
    [checkpoint, modelName, description, license, outputDir, repoId, hfToken, isPrivate]
  )

  const canPush = checkpoint && repoId.includes('/')
  const canLocal = checkpoint && outputDir

  return (
    <div className="max-w-2xl space-y-6">
      <p className="text-sm text-slate-500">
        Package a trained checkpoint in HuggingFace format — loadable with{' '}
        <code className="text-slate-400">AutoModelForCausalLM.from_pretrained(…, trust_remote_code=True)</code>.
        The export bundles custom model code so the published model uses the saved graph and weights; see the capability matrix for validated numerical tolerances.
      </p>

      <Field label="Checkpoint" help="Checkpoints from the selected registered run, including custom folders">
        <CheckpointSelect checkpoints={checkpoints} value={checkpoint} onChange={setCheckpoint} />
      </Field>

      {detecting && (
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Loader2 size={14} className="animate-spin" /> Analyzing checkpoint…
        </div>
      )}
      {detection && (
        <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 text-sm space-y-1">
          <div className="flex gap-2">
            <span className="text-slate-500 w-20 flex-none">Format</span>
            <span className="text-slate-200 font-medium">{detection.format}</span>
          </div>
          <div className="flex gap-2">
            <span className="text-slate-500 w-20 flex-none">Why</span>
            <span className="text-slate-400">{detection.reason}</span>
          </div>
        </div>
      )}

      <div className="space-y-3">
        <Field label="Model name" help="Display name used in the model card (optional)">
          <input className={inputCls} value={modelName} onChange={(e) => setModelName(e.target.value)} placeholder="My tiny LLM" />
        </Field>
        <Field label="Description" help="One or two sentences for the README (optional)">
          <textarea
            className={inputCls + ' resize-y min-h-[60px]'}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="A 50M-parameter model trained on FineWeb-Edu…"
          />
        </Field>
        <Field label="License">
          <select className={inputCls + ' appearance-none'} value={license} onChange={(e) => setLicense(e.target.value)}>
            {['apache-2.0', 'mit', 'cc-by-4.0', 'cc-by-nc-4.0', 'openrail', 'other'].map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </Field>
      </div>

      {/* Local export */}
      <div className="border border-slate-800 rounded-xl p-4 space-y-3">
        <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <Upload size={15} className="text-accent" /> Export to folder
        </h3>
        <Field label="Output directory" help="A ready-to-share folder is written here">
          <input className={inputCls + ' font-mono'} value={outputDir} onChange={(e) => setOutputDir(e.target.value)} />
        </Field>
        <button
          onClick={() => doExport('local')}
          disabled={!canLocal || busy !== null}
          className="px-4 py-2 text-sm rounded-lg font-medium bg-accent/10 text-accent border border-accent/20 hover:bg-accent/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {busy === 'local' ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
          Export locally
        </button>
      </div>

      {/* Hub push */}
      <div className="border border-slate-800 rounded-xl p-4 space-y-3">
        <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <Rocket size={15} className="text-amber-400" /> Push to Hugging Face Hub
        </h3>
        <Field label="Repository" help="username/model-name — created if it doesn't exist">
          <input className={inputCls + ' font-mono'} value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="username/my-tiny-llm" />
        </Field>
        <Field label="Access token" help="Optional with backend hf login; otherwise use a write token — never stored">
          <input type="password" className={inputCls + ' font-mono'} value={hfToken} onChange={(e) => setHfToken(e.target.value)} placeholder="hf_…" />
        </Field>
        <Field label="Private repo">
          <button
            type="button"
            role="switch" aria-checked={isPrivate} aria-label="Private repository"
            onClick={() => setIsPrivate(!isPrivate)}
            className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${isPrivate ? 'bg-accent' : 'bg-slate-700'}`}
          >
            <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${isPrivate ? 'translate-x-4' : 'translate-x-0'}`} />
          </button>
        </Field>
        <button
          onClick={() => doExport('hub')}
          disabled={!canPush || busy !== null}
          className="px-4 py-2 text-sm rounded-lg font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 hover:bg-amber-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
        >
          {busy === 'hub' ? <Loader2 size={14} className="animate-spin" /> : <Rocket size={14} />}
          Push to Hub
        </button>
      </div>

      <ResultBanner result={result} />
      {result?.success && result.hub_url && (
        <a
          href={result.hub_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-sm text-accent hover:underline"
        >
          <ExternalLink size={14} /> {result.hub_url}
        </a>
      )}
      {result?.success && result.files && (
        <div className="text-xs text-slate-500">
          Files: {result.files.join(', ')}
        </div>
      )}
    </div>
  )
}

// ─── Playground tab ───────────────────────────────────────────────────────────

function PlaygroundTab({ runId }: { runId: string }) {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [checkpoint, setCheckpoint] = useState('')
  const [prompt, setPrompt] = useState('')
  const [maxTokens, setMaxTokens] = useState(128)
  const [temperature, setTemperature] = useState(0.8)
  const [topP, setTopP] = useState(0.9)
  const [busy, setBusy] = useState(false)
  const [output, setOutput] = useState<{ text: string; tokens: number; tps: number } | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    listCheckpoints(runId).then(setCheckpoints).catch(() => setCheckpoints([]))
  }, [runId])

  const generate = useCallback(async () => {
    setBusy(true)
    setError('')
    setOutput(null)
    try {
      const res = await generateText({
        checkpoint_path: checkpoint,
        prompt,
        max_tokens: maxTokens,
        temperature,
        top_p: topP,
      })
      setOutput({ text: res.generated_text, tokens: res.tokens_generated, tps: res.tokens_per_second })
    } catch (err: unknown) {
      setError(errMessage(err))
    } finally {
      setBusy(false)
    }
  }, [checkpoint, prompt, maxTokens, temperature, topP])

  return (
    <div className="max-w-2xl space-y-6">
      <p className="text-sm text-slate-500">
        Chat with a checkpoint straight from the lab. Base (pretrain-only) models continue text; SFT/RLHF models follow instructions.
      </p>

      <Field label="Checkpoint">
        <CheckpointSelect checkpoints={checkpoints} value={checkpoint} onChange={setCheckpoint} />
      </Field>

      <Field label="Prompt">
        <textarea
          className={inputCls + ' resize-y min-h-[100px]'}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Once upon a time…"
        />
      </Field>

      <div className="grid grid-cols-3 gap-4">
        <Field label="" help={`max tokens: ${maxTokens}`}>
          <input type="range" min={16} max={1024} step={16} value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} className="w-full accent-emerald-500" />
        </Field>
        <Field label="" help={`temperature: ${temperature.toFixed(2)}`}>
          <input type="range" min={0} max={2} step={0.05} value={temperature} onChange={(e) => setTemperature(+e.target.value)} className="w-full accent-emerald-500" />
        </Field>
        <Field label="" help={`top_p: ${topP.toFixed(2)}`}>
          <input type="range" min={0.1} max={1} step={0.05} value={topP} onChange={(e) => setTopP(+e.target.value)} className="w-full accent-emerald-500" />
        </Field>
      </div>

      <button
        onClick={generate}
        disabled={!checkpoint || !prompt || busy}
        className="px-4 py-2 text-sm rounded-lg font-medium bg-accent/10 text-accent border border-accent/20 hover:bg-accent/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
      >
        {busy ? <Loader2 size={14} className="animate-spin" /> : <MessageSquare size={14} />}
        Generate
      </button>

      {error && <ResultBanner result={{ success: false, message: error }} />}
      {output && (
        <div className="space-y-2">
          <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700 text-sm text-slate-200 whitespace-pre-wrap">
            {output.text}
          </div>
          <p className="text-xs text-slate-600">
            {output.tokens} tokens · {output.tps.toFixed(1)} tok/s
            {output.tps > 0 && ` · ${(output.tokens / output.tps).toFixed(1)}s`}
          </p>
        </div>
      )}
    </div>
  )
}

// ─── Merge tab ────────────────────────────────────────────────────────────────

function MergeTab() {
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [adapters, setAdapters] = useState<AdapterInfo[]>([])
  const [mode, setMode] = useState<'adapter_folder' | 'full_checkpoint'>('adapter_folder')
  const [base, setBase] = useState('')
  const [adapter, setAdapter] = useState('')
  const [outputPath, setOutputPath] = useState('/app/data/merged_model.pt')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null)

  useEffect(() => {
    listMergeAvailable()
      .then(({ checkpoints, adapters }) => {
        setCheckpoints(checkpoints)
        setAdapters(adapters)
      })
      .catch(() => {})
  }, [])

  const doMerge = useCallback(async () => {
    setBusy(true)
    setResult(null)
    try {
      const res = await mergeLora({
        adapter_path: mode === 'adapter_folder' ? adapter : base,
        base_checkpoint_path: base,
        output_path: outputPath,
        input_type: mode,
      })
      setResult(res)
    } catch (err: unknown) {
      setResult({ success: false, message: errMessage(err) })
    } finally {
      setBusy(false)
    }
  }, [mode, adapter, base, outputPath])

  const ready = outputPath && base && (mode === 'full_checkpoint' || adapter)

  return (
    <div className="max-w-2xl space-y-6">
      <p className="text-sm text-slate-500">
        Fold LoRA adapters back into the base model to get a single standalone checkpoint (required before export).
      </p>

      <Field label="Source" help="Adapter folder = LoRA saved separately · Full checkpoint = LoRA weights inside the checkpoint">
        <select className={inputCls + ' appearance-none'} value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
          <option value="adapter_folder">Base checkpoint + adapter folder</option>
          <option value="full_checkpoint">Checkpoint containing LoRA weights</option>
        </select>
      </Field>

      <Field label={mode === 'adapter_folder' ? 'Base checkpoint' : 'Checkpoint'}>
        <CheckpointSelect checkpoints={checkpoints} value={base} onChange={setBase} />
      </Field>

      {mode === 'adapter_folder' && (
        <Field label="LoRA adapters">
          <select className={inputCls + ' appearance-none'} value={adapter} onChange={(e) => setAdapter(e.target.value)}>
            <option value="">Select adapters…</option>
            {adapters.map((a) => (
              <option key={a.path} value={a.path}>
                {a.name} · r={a.rank} α={a.alpha}
              </option>
            ))}
          </select>
        </Field>
      )}

      <Field label="Output path">
        <input className={inputCls + ' font-mono'} value={outputPath} onChange={(e) => setOutputPath(e.target.value)} />
      </Field>

      <button
        onClick={doMerge}
        disabled={!ready || busy}
        className="px-4 py-2 text-sm rounded-lg font-medium bg-accent/10 text-accent border border-accent/20 hover:bg-accent/20 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
      >
        {busy ? <Loader2 size={14} className="animate-spin" /> : <GitMerge size={14} />}
        Merge
      </button>

      <ResultBanner result={result} />
    </div>
  )
}

// ─── ToolsView ────────────────────────────────────────────────────────────────

type Tab = 'export' | 'playground' | 'merge'

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'export', label: 'Export & Push', icon: <Rocket size={14} /> },
  { id: 'playground', label: 'Playground', icon: <MessageSquare size={14} /> },
  { id: 'merge', label: 'Merge LoRA', icon: <GitMerge size={14} /> },
]

interface Props {
  onBack: () => void
  initialTab?: Tab
}

export default function ToolsView({ onBack, initialTab = 'export' }: Props) {
  const [tab, setTab] = useState<Tab>(initialTab)
  const [runId, setRunId] = useState('')
  const [runs, setRuns] = useState<TrainingStatus[]>([])
  useEffect(() => { listRuns().then(setRuns).catch(() => setRuns([])) }, [])

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur flex-none">
        <div className="px-6 py-3 flex items-center gap-4">
          <button onClick={onBack} className="flex items-center gap-1.5 text-slate-500 hover:text-slate-300 transition-colors text-sm">
            <ArrowLeft size={16} />
            Projects
          </button>
          <div className="w-px h-4 bg-slate-700" />
          <FlaskConical size={16} className="text-accent" />
          <span className="font-semibold text-slate-100">Model Tools</span>
        </div>
      </header>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <aside className="w-52 flex-none bg-slate-900/50 border-r border-slate-800 px-4 py-6 space-y-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                tab === t.id ? 'bg-accent/10 text-accent' : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </aside>

        <main className="flex-1 min-w-0 overflow-auto p-8">
          {tab !== 'merge' && <label className="mb-6 block max-w-2xl text-sm text-slate-400">Recorded run
            <select aria-label="Tools recorded run" className={inputCls + ' mt-2'} value={runId} onChange={e => setRunId(e.target.value)}>
              <option value="">All registered and legacy checkpoints</option>
              {runs.map(r => <option key={r.run_id} value={r.run_id ?? ''}>{r.run_id} · {r.training_type} · {r.status} · step {r.current_step}</option>)}
            </select>
          </label>}
          {tab === 'export' && <ExportTab key={runId} runId={runId} />}
          {tab === 'playground' && <PlaygroundTab key={runId} runId={runId} />}
          {tab === 'merge' && <MergeTab />}
        </main>
      </div>
    </div>
  )
}
