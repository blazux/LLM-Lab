import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Square, Cpu, Database, Settings, Activity, AlertCircle } from 'lucide-react'
import type { Project, TrainingStatus, MetricPoint, SFTConfig, RLHFConfig, ModelConfig, TrainingConfig } from '../../types'
import {
  startPretraining,
  startSFT,
  startRLHF,
  stopTraining,
  getTrainingStatus,
  listRuns,
  getRun,
  importRun,
  resumeRun,
  createMetricsStream,
  buildPretrainPayload,
  buildSFTPayload,
  buildRLHFPayload,
  errMessage,
  sanitizeModelConfig,
} from '../../services/api'
import MetricsPanel from '../MetricsPanel'
import { finiteMetric, mergeMetricEvent } from '../../utils/metrics'
import { estimateParams } from '../../utils/estimate'
import { useProjects } from '../../context/ProjectContext'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

function formatNumber(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return n.toString()
}

// ─── Summary card row ─────────────────────────────────────────────────────────

function SummaryRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div className="flex items-center gap-3 py-2">
      <div className="text-slate-500 flex-none">{icon}</div>
      <span className="text-sm text-slate-500 flex-1">{label}</span>
      <span className="text-sm text-slate-200 font-medium">{value}</span>
    </div>
  )
}

// ─── Log line ─────────────────────────────────────────────────────────────────

interface LogLine {
  id: number
  ts: string
  msg: string
  type: 'info' | 'success' | 'error' | 'metric'
}

// ─── RunStep ──────────────────────────────────────────────────────────────────

interface Props {
  project: Project
}

export default function RunStep({ project }: Props) {
  const { updateProject } = useProjects()
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [selectedRunId, setSelectedRunId] = useState(project.lastRunId ?? '')
  const [runs, setRuns] = useState<TrainingStatus[]>([])
  const [activeStatus, setActiveStatus] = useState<TrainingStatus | null>(null)
  const [showAllRuns, setShowAllRuns] = useState(false)
  const [importFolder, setImportFolder] = useState('')
  const [additionalSteps, setAdditionalSteps] = useState(0)
  const autoSelect = useRef(true)
  const seenEvents = useRef(new Set<number>())
  const [metrics, setMetrics] = useState<MetricPoint[]>([])
  const [logs, setLogs] = useState<LogLine[]>([])
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)
  const logIdRef = useRef(0)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const addLog = useCallback((msg: string, type: LogLine['type'] = 'info', timestamp?: number) => {
    const id = logIdRef.current++
    const ts = new Date(timestamp !== undefined ? timestamp * 1000 : Date.now()).toLocaleTimeString()
    setLogs((prev) => [...prev.slice(-499), { id, ts, msg, type }])
  }, [])

  // Selection is persistent on the project; the server owns actual run configuration.
  useEffect(() => {
    setSelectedRunId(project.lastRunId ?? '')
    autoSelect.current = true
  }, [project.id])

  useEffect(() => {
    let cancelled = false
    async function poll() {
      while (!cancelled) {
        try {
          const [active, records] = await Promise.all([getTrainingStatus(), listRuns()])
          const selected = selectedRunId ? await getRun(selectedRunId) : null
          if (cancelled) return
          setActiveStatus(active)
          setRuns(records)
          setStatus(selected)
          if (!selectedRunId && autoSelect.current) {
            autoSelect.current = false
            const latest = records.find(r => r.project_id === project.id)
            if (latest?.run_id) setSelectedRunId(latest.run_id)
          }
        } catch {
          // Preserve existing history while the backend is temporarily unavailable.
        }
        await new Promise(r => setTimeout(r, 2000))
      }
    }
    poll()
    return () => { cancelled = true }
  }, [selectedRunId, project.id])

  // SSE stream
  useEffect(() => {
    setMetrics([])
    setLogs([])
    seenEvents.current.clear()
    if (!selectedRunId) return
    const es = createMetricsStream(
      (data) => {
        if (data.run_id !== selectedRunId) return
        if (data.type === 'snapshot' || data.type === 'status') {
          setStatus(data as unknown as TrainingStatus)
          return
        }
        const eventId = finiteMetric(data.event_id)
        if (eventId !== undefined) {
          if (seenEvents.current.has(eventId)) return
          seenEvents.current.add(eventId)
        }
        // Backend log lines (including "Training failed: ...")
        if (data.type === 'log' && typeof data.message === 'string') {
          const level = data.level === 'error' ? 'error' : data.level === 'success' ? 'success' : 'info'
          addLog(data.message, level, finiteMetric(data.timestamp))
          return
        }
        setMetrics(prev => mergeMetricEvent(prev, data))
        if (data.type === 'eval_metrics') {
          const l = finiteMetric(data.eval_loss)
          const p = finiteMetric(data.eval_perplexity)
          if (l !== undefined) addLog(`eval  loss=${l.toFixed(4)}${p ? `  ppl=${p.toFixed(1)}` : ''}`, 'success')
          return
        }

        const step = finiteMetric(data.step)
        const loss = finiteMetric(data.loss)
        const lr = finiteMetric(data.lr)
        const tps = finiteMetric(data.tokens_per_second)

        if (step !== undefined && loss !== undefined) {
          addLog(`step=${step}  loss=${loss.toFixed(4)}${lr ? `  lr=${lr.toExponential(2)}` : ''}${tps ? `  tok/s=${Math.round(tps)}` : ''}`, 'metric')
        }
      },
      () => {
        // EventSource reconnects with Last-Event-ID; the server replays missing events.
      },
      selectedRunId
    )
    esRef.current = es
    return () => {
      es.close()
      esRef.current = null
    }
  }, [addLog, selectedRunId])

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  async function handleStart() {
    setError(null)
    setIsStarting(true)
    addLog('Sending start request to backend…')
    try {
      const modelCfg = { ...project.modelConfig } as unknown as Record<string, unknown>
      const trainCfg = project.trainingConfig

      if (project.type === 'pretrain') {
        if (project.modelConfig.assembly) {
          const tokenizerResponse = await fetch('/api/assembly/tokenizer', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: project.modelConfig.tokenizer_name }) })
          const tokenizer = await tokenizerResponse.json()
          if (!tokenizerResponse.ok) throw new Error(typeof tokenizer.detail === 'string' ? tokenizer.detail : 'Could not read tokenizer.')
          modelCfg.vocab_size = tokenizer.vocab_size
          updateProject(project.id, { modelConfig: { ...project.modelConfig, vocab_size: tokenizer.vocab_size } })
          const response = await fetch('/api/assembly/inspect', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_cfg: sanitizeModelConfig(modelCfg) }) })
          if (!response.ok) throw new Error('Could not validate the graph before training.')
          const inspection = await response.json()
          if (!inspection.valid) throw new Error(inspection.errors?.[0]?.message ?? 'Fix the graph before training.')
        }
        const payload = buildPretrainPayload(modelCfg, trainCfg)
        const started = await startPretraining({ ...payload, project_id: project.id })
        selectRun(started.run_id)
        addLog('Pretraining started', 'success')
      } else if (project.type === 'sft') {
        const payload = buildSFTPayload(modelCfg, trainCfg as SFTConfig)
        const started = await startSFT({ ...payload, project_id: project.id })
        selectRun(started.run_id)
        addLog('SFT training started', 'success')
      } else if (project.type === 'rlhf') {
        const cfg = trainCfg as RLHFConfig
        if (!cfg.policy_checkpoint) {
          setError('Set policy_checkpoint in the Training step — RLHF starts from an SFT (or pretrained) checkpoint.')
          addLog('RLHF start blocked: no policy checkpoint', 'error')
          return
        }
        if (cfg.batch_size % (cfg.mini_batch_size || 1) !== 0) {
          setError(`batch_size (${cfg.batch_size}) must be divisible by mini_batch_size (${cfg.mini_batch_size}).`)
          addLog('RLHF start blocked: batch_size not divisible by mini_batch_size', 'error')
          return
        }
        const started = await startRLHF({ ...buildRLHFPayload(cfg), project_id: project.id })
        selectRun(started.run_id)
        addLog(`${cfg.algorithm.toUpperCase()} training started`, 'success')
      }
    } catch (err: unknown) {
      const msg = errMessage(err)
      setError(msg)
      addLog(`Failed to start: ${msg}`, 'error')
    } finally {
      setIsStarting(false)
    }
  }

  async function handleStop() {
    setIsStopping(true)
    addLog('Stopping training…')
    try {
      await stopTraining(selectedRunId)
      addLog('Stop requested; finishing the current update and saving a checkpoint.', 'info')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      addLog(`Failed to stop: ${msg}`, 'error')
    } finally {
      setIsStopping(false)
    }
  }

  function selectRun(runId: string) {
    autoSelect.current = false
    setStatus(null)
    setSelectedRunId(runId)
    updateProject(project.id, { lastRunId: runId })
  }

  async function handleImport() {
    setError(null)
    try { selectRun((await importRun(importFolder, project.id)).run_id) }
    catch (error) { setError(errMessage(error)) }
  }

  async function handleResume() {
    setIsStarting(true); setError(null)
    try { selectRun((await resumeRun(selectedRunId, additionalSteps)).run_id) }
    catch (error) { setError(errMessage(error)) }
    finally { setIsStarting(false) }
  }

  const displayedTraining = status?.training_config ? status.training_config as unknown as TrainingConfig : project.trainingConfig
  const displayedModel = status?.model_config ? { ...status.model_config,
    architecture: status.model_config.model_architecture ?? status.model_config.architecture ?? 'transformer' } as unknown as ModelConfig : project.modelConfig
  const selectedType = status?.training_type === 'pretraining' ? 'pretrain' : status?.training_type ?? project.type
  const visibleRuns = runs.filter(r => showAllRuns || r.project_id === project.id || !r.project_id || r.run_id === selectedRunId)
  const isTraining = status?.is_training ?? false
  const currentStep = status?.current_step ?? 0
  const totalSteps = (status?.max_steps || displayedTraining.max_steps) ?? 0
  const progress = totalSteps > 0 ? Math.min((currentStep / totalSteps) * 100, 100) : 0
  const effectiveBatch = displayedTraining.batch_size * displayedTraining.gradient_accumulation_steps

  return (
    <div className="grid grid-cols-2 gap-6 h-full min-h-0">
      {/* Left column: summary + controls */}
      <div className="flex flex-col gap-4 overflow-y-auto">
        <section className="space-y-3 rounded-xl border border-slate-700 bg-slate-900 p-4">
          <label className="block text-xs text-slate-400">Recorded run
            <select aria-label="Recorded run" value={selectedRunId} onChange={e => selectRun(e.target.value)} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 p-2 text-sm text-slate-200">
              <option value="">New run / project settings</option>
              {visibleRuns.map(r => <option key={r.run_id} value={r.run_id ?? ''}>{r.created_at ? new Date(r.created_at).toLocaleString() : r.run_id} · {r.training_type} · {r.status} · step {r.current_step}</option>)}
            </select>
          </label>
          <label className="flex gap-2 text-xs text-slate-400"><input type="checkbox" checked={showAllRuns} onChange={e => setShowAllRuns(e.target.checked)} />Show runs from all projects</label>
          {status?.output_dir && <p className="break-all text-xs text-slate-400">Saved output: {status.output_dir}</p>}
          {status?.run_id && <p className="break-all text-xs text-slate-500">Run ID: {status.run_id} · {status.status}</p>}
          <details><summary className="cursor-pointer text-xs text-accent">Import an existing pretraining run</summary>
            <input aria-label="Existing run folder" className="mt-2 w-full rounded bg-slate-800 p-2 text-sm text-slate-200" placeholder="/path/to/run" value={importFolder} onChange={e => setImportFolder(e.target.value)} />
            <button className="mt-2 text-xs text-accent" disabled={!importFolder} onClick={handleImport}>Import checkpoint folder</button>
            <p className="mt-1 text-xs text-slate-500">Imports the saved configuration and final evaluation. Earlier curves cannot be reconstructed if metric events were never saved.</p>
          </details>
          {status?.run_id && !status.is_training && status.training_type === 'pretraining' && <div className="space-y-2 border-t border-slate-700 pt-2">
            <label className="block text-xs text-slate-400">Additional optimizer steps (0 continues the saved budget)<input aria-label="Additional optimizer steps" type="number" min={0} step={1} value={additionalSteps} onChange={e => setAdditionalSteps(Number(e.target.value))} className="mt-1 w-full rounded bg-slate-800 p-2 text-sm text-slate-200" /></label>
            <button onClick={handleResume} disabled={isStarting || !!activeStatus?.is_training} className="text-xs text-accent disabled:opacity-40">Resume saved checkpoint into a new run folder</button>
          </div>}
          {activeStatus?.is_training && activeStatus.run_id !== selectedRunId && <p className="text-xs text-amber-200">Another run is active. Select it to monitor or stop it.</p>}
        </section>
        {/* Project summary */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">{status?.run_id ? 'Saved run configuration' : 'Project settings for a new run'}</h4>
          <div className="divide-y divide-slate-800">
            <SummaryRow icon={<Cpu size={14} />} label="Architecture" value={status?.run_id && !status.model_config ? 'From policy checkpoint' : displayedModel.assembly ? 'Visual assembly' : displayedModel.architecture} />
            <SummaryRow icon={<Cpu size={14} />} label="Parameters" value={status?.run_id && !status.model_config ? 'Not recorded' : `~${formatNumber(estimateParams(displayedModel))}`} />
            <SummaryRow icon={<Settings size={14} />} label="Optimizer" value={displayedTraining.optimizer.toUpperCase()} />
            {selectedType === 'rlhf' ? (
              <SummaryRow icon={<Settings size={14} />} label="Algorithm" value={((displayedTraining as RLHFConfig).algorithm ?? 'grpo').toUpperCase()} />
            ) : (
              <SummaryRow icon={<Settings size={14} />} label="Scheduler" value={displayedTraining.scheduler} />
            )}
            <SummaryRow icon={<Database size={14} />} label="Datasets" value={displayedTraining.datasets.length} />
            <SummaryRow icon={<Activity size={14} />} label="Max steps" value={displayedTraining.max_steps.toLocaleString()} />
            <SummaryRow icon={<Activity size={14} />} label="Effective batch" value={effectiveBatch} />
            {selectedType !== 'rlhf' && (
              <SummaryRow icon={<Activity size={14} />} label="Loss fn" value={displayedTraining.loss_fn} />
            )}
          </div>
        </div>

        {/* Status panel */}
        {status && (
          <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className={`w-2 h-2 rounded-full ${isTraining ? 'bg-accent animate-pulse' : 'bg-slate-600'}`} />
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                {status.status}
              </span>
            </div>

            {/* Progress bar */}
            <div className="mb-3">
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>Step {currentStep.toLocaleString()}</span>
                <span>{totalSteps.toLocaleString()} total</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-1.5">
                <motion.div
                  className="bg-accent h-1.5 rounded-full"
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <div className="text-right text-xs text-slate-600 mt-0.5">{progress.toFixed(1)}%</div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-3 gap-2">
              {status.current_loss != null && (
                <div className="bg-slate-900 rounded-lg p-2 text-center">
                  <div className="text-xs text-slate-500">Objective loss</div>
                  <div className="text-sm font-semibold text-slate-200">{status.current_loss.toFixed(4)}</div>
                </div>
              )}
              {status.current_ppl != null && (
                <div className="bg-slate-900 rounded-lg p-2 text-center">
                  <div className="text-xs text-slate-500">Reported perplexity</div>
                  <div className="text-sm font-semibold text-slate-200">{status.current_ppl.toFixed(1)}</div>
                </div>
              )}
              {status.current_lr != null && (
                <div className="bg-slate-900 rounded-lg p-2 text-center">
                  <div className="text-xs text-slate-500">LR</div>
                  <div className="text-sm font-semibold text-slate-200">{status.current_lr.toExponential(2)}</div>
                </div>
              )}
            </div>

            {/* Backend-reported failure */}
            {status.status === 'error' && status.error && (
              <div className="mt-3 flex items-start gap-2 bg-red-500/10 border border-red-500/30 rounded-lg p-2.5">
                <AlertCircle size={13} className="text-red-400 flex-none mt-0.5" />
                <p className="text-xs text-red-400 break-all">{status.error}</p>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-start gap-2 bg-red-500/10 border border-red-500/30 rounded-xl p-3"
            >
              <AlertCircle size={14} className="text-red-400 flex-none mt-0.5" />
              <p className="text-xs text-red-400">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Controls */}
        <div className="flex gap-3">
          {!isTraining ? (
            <button
              onClick={handleStart}
              disabled={isStarting || !!activeStatus?.is_training}
              className="flex-1 flex items-center justify-center gap-2 py-3 bg-accent hover:bg-accent-dim text-slate-950 font-semibold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Play size={16} />
              {isStarting ? 'Starting…' : 'Start new training from project settings'}
            </button>
          ) : (
            <button
              onClick={handleStop}
              disabled={isStopping}
              className="flex-1 flex items-center justify-center gap-2 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 font-semibold rounded-xl transition-colors disabled:opacity-50"
            >
              <Square size={16} />
              {isStopping ? 'Stopping…' : 'Stop Training'}
            </button>
          )}
        </div>


      </div>

      {/* Right column: metrics + logs */}
      <div className="flex flex-col gap-4 min-h-0 overflow-y-auto">
        {/* Charts */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Metrics</h4>
          <MetricsPanel metrics={metrics} metricDefinition={status?.metric_definition} />
        </div>

        {/* Logs */}
        <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
            <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">Event Log</span>
            <button
              onClick={() => setLogs([])}
              className="text-xs text-slate-600 hover:text-slate-400 transition-colors"
            >
              Clear
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-1">
            {logs.length === 0 && (
              <div className="text-slate-700 italic">No events yet</div>
            )}
            {logs.map((line) => (
              <div key={line.id} className={`flex gap-2 ${
                line.type === 'error' ? 'text-red-400' :
                line.type === 'success' ? 'text-accent' :
                line.type === 'metric' ? 'text-slate-400' :
                'text-slate-500'
              }`}>
                <span className="text-slate-700 flex-none">{line.ts}</span>
                <span>{line.msg}</span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    </div>
  )
}
