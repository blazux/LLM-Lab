import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Square, Cpu, Database, Settings, Activity, AlertCircle, CheckCircle2, Clock } from 'lucide-react'
import type { Project, TrainingStatus, MetricPoint, SFTConfig } from '../../types'
import {
  startPretraining,
  startSFT,
  startRLHF,
  stopTraining,
  getTrainingStatus,
  createMetricsStream,
  buildPretrainPayload,
  buildSFTPayload,
} from '../../services/api'
import MetricsPanel from '../MetricsPanel'

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
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [metrics, setMetrics] = useState<MetricPoint[]>([])
  const [logs, setLogs] = useState<LogLine[]>([])
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)
  const logIdRef = useRef(0)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const addLog = useCallback((msg: string, type: LogLine['type'] = 'info') => {
    const id = logIdRef.current++
    const ts = new Date().toLocaleTimeString()
    setLogs((prev) => [...prev.slice(-499), { id, ts, msg, type }])
  }, [])

  // Poll status every 2s
  useEffect(() => {
    let cancelled = false
    async function poll() {
      while (!cancelled) {
        try {
          const s = await getTrainingStatus()
          setStatus(s)
        } catch {
          // backend not running
        }
        await new Promise((r) => setTimeout(r, 2000))
      }
    }
    poll()
    return () => { cancelled = true }
  }, [])

  // SSE stream
  useEffect(() => {
    const es = createMetricsStream(
      (data) => {
        const step = data.step as number | undefined
        const loss = data.loss as number | undefined
        const lr = data.lr as number | undefined
        const tps = data.tokens_per_second as number | undefined

        if (step !== undefined && loss !== undefined) {
          setMetrics((prev) => {
            // Deduplicate by step
            const filtered = prev.filter((m) => m.step !== step)
            return [...filtered, { step, loss, lr, tokens_per_second: tps }].sort((a, b) => a.step - b.step)
          })
          addLog(`step=${step}  loss=${loss.toFixed(4)}${lr ? `  lr=${lr.toExponential(2)}` : ''}${tps ? `  tok/s=${Math.round(tps)}` : ''}`, 'metric')
        }
      },
      () => {
        // SSE connection error — not fatal (training might not be running)
      }
    )
    esRef.current = es
    return () => {
      es.close()
      esRef.current = null
    }
  }, [addLog])

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  async function handleStart() {
    setError(null)
    setIsStarting(true)
    addLog('Sending start request to backend…')
    try {
      const modelCfg = project.modelConfig as unknown as Record<string, unknown>
      const trainCfg = project.trainingConfig

      if (project.type === 'pretrain') {
        const payload = buildPretrainPayload(modelCfg, trainCfg)
        await startPretraining(payload)
        addLog('Pretraining started', 'success')
      } else if (project.type === 'sft') {
        const payload = buildSFTPayload(modelCfg, trainCfg as SFTConfig)
        await startSFT(payload)
        addLog('SFT training started', 'success')
      } else if (project.type === 'rlhf') {
        const payload = buildPretrainPayload(modelCfg, trainCfg)
        await startRLHF(payload)
        addLog('RLHF training started', 'success')
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
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
      await stopTraining()
      addLog('Training stopped', 'success')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      addLog(`Failed to stop: ${msg}`, 'error')
    } finally {
      setIsStopping(false)
    }
  }

  const isTraining = status?.is_training ?? false
  const currentStep = status?.current_step ?? 0
  const totalSteps = status?.total_steps ?? project.trainingConfig.max_steps ?? 0
  const progress = totalSteps > 0 ? Math.min((currentStep / totalSteps) * 100, 100) : 0
  const effectiveBatch = project.trainingConfig.batch_size * project.trainingConfig.gradient_accumulation_steps

  return (
    <div className="grid grid-cols-2 gap-6 h-full min-h-0">
      {/* Left column: summary + controls */}
      <div className="flex flex-col gap-4 overflow-y-auto">
        {/* Project summary */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Project Summary</h4>
          <div className="divide-y divide-slate-800">
            <SummaryRow icon={<Cpu size={14} />} label="Architecture" value={project.modelConfig.architecture} />
            <SummaryRow icon={<Cpu size={14} />} label="Parameters" value={`${formatNumber(project.modelConfig.d_model * project.modelConfig.n_layers)}+`} />
            <SummaryRow icon={<Settings size={14} />} label="Optimizer" value={project.trainingConfig.optimizer.toUpperCase()} />
            <SummaryRow icon={<Settings size={14} />} label="Scheduler" value={project.trainingConfig.scheduler} />
            <SummaryRow icon={<Database size={14} />} label="Datasets" value={project.trainingConfig.datasets.length} />
            <SummaryRow icon={<Activity size={14} />} label="Max steps" value={project.trainingConfig.max_steps.toLocaleString()} />
            <SummaryRow icon={<Activity size={14} />} label="Effective batch" value={effectiveBatch} />
            <SummaryRow icon={<Activity size={14} />} label="Loss fn" value={project.trainingConfig.loss_fn} />
          </div>
        </div>

        {/* Status panel */}
        {status && (
          <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className={`w-2 h-2 rounded-full ${isTraining ? 'bg-accent animate-pulse' : 'bg-slate-600'}`} />
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                {isTraining ? 'Training' : 'Idle'}
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
            <div className="grid grid-cols-2 gap-2">
              {status.loss != null && (
                <div className="bg-slate-900 rounded-lg p-2 text-center">
                  <div className="text-xs text-slate-500">Loss</div>
                  <div className="text-sm font-semibold text-slate-200">{status.loss.toFixed(4)}</div>
                </div>
              )}
              {status.lr != null && (
                <div className="bg-slate-900 rounded-lg p-2 text-center">
                  <div className="text-xs text-slate-500">LR</div>
                  <div className="text-sm font-semibold text-slate-200">{status.lr.toExponential(2)}</div>
                </div>
              )}
              {status.tokens_per_second != null && (
                <div className="bg-slate-900 rounded-lg p-2 text-center">
                  <div className="text-xs text-slate-500">tok/s</div>
                  <div className="text-sm font-semibold text-slate-200">{Math.round(status.tokens_per_second)}</div>
                </div>
              )}
              {status.elapsed_time != null && (
                <div className="bg-slate-900 rounded-lg p-2 text-center">
                  <div className="text-xs text-slate-500">Elapsed</div>
                  <div className="text-sm font-semibold text-slate-200">{formatDuration(status.elapsed_time)}</div>
                </div>
              )}
              {status.estimated_remaining != null && status.estimated_remaining > 0 && (
                <div className="bg-slate-900 rounded-lg p-2 text-center col-span-2">
                  <div className="text-xs text-slate-500 flex items-center justify-center gap-1"><Clock size={10} /> ETA</div>
                  <div className="text-sm font-semibold text-slate-200">{formatDuration(status.estimated_remaining)}</div>
                </div>
              )}
            </div>
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
              disabled={isStarting}
              className="flex-1 flex items-center justify-center gap-2 py-3 bg-accent hover:bg-accent-dim text-slate-950 font-semibold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Play size={16} />
              {isStarting ? 'Starting…' : 'Start Training'}
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

        {/* Checkpoint info */}
        {status?.checkpoint_path && (
          <div className="flex items-center gap-2 bg-accent/5 border border-accent/20 rounded-xl p-3">
            <CheckCircle2 size={14} className="text-accent flex-none" />
            <div>
              <div className="text-xs text-slate-400 font-medium">Latest checkpoint</div>
              <div className="text-xs text-slate-500 font-mono mt-0.5">{status.checkpoint_path}</div>
            </div>
          </div>
        )}
      </div>

      {/* Right column: metrics + logs */}
      <div className="flex flex-col gap-4 min-h-0 overflow-hidden">
        {/* Charts */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Metrics</h4>
          <MetricsPanel metrics={metrics} />
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
