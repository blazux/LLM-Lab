import { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, Square, Terminal, TrendingDown, Zap, Clock, CheckCircle, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import { useTraining, TrainingLog } from '../context/TrainingContext';
import { subscribeToMetrics, TrainingMetric, stopTraining } from '../services/trainingApi';

const TrainingMonitor = () => {
  const { trainingState, updateTrainingState, addMetric, updateMetricWithEval, addLog } = useTraining();
  const { isTraining, progress, currentStep, maxSteps, currentLoss, currentPPL, currentLR, metrics, logs } = trainingState;

  const [eta, setEta] = useState('--:--:--');
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const lastStepRef = useRef<number>(0);
  const recentStepsRef = useRef<Array<{ step: number; time: number }>>([]);

  // Subscribe to real-time metrics when training starts
  useEffect(() => {
    if (isTraining) {
      // Subscribe to metrics stream
      const unsubscribe = subscribeToMetrics(
        (metric: TrainingMetric) => {
          if (metric.type === 'snapshot') {
            // Handle snapshot (reconnect state)
            console.log('Received snapshot, restoring state:', metric);

            // Update training state with current server state
            updateTrainingState({
              currentStep: metric.current_step ?? 0,
              currentLoss: metric.current_loss ?? null,
              currentPPL: metric.current_ppl ?? null,
              currentLR: metric.current_lr ?? null,
              maxSteps: metric.max_steps ?? maxSteps,
              progress: (metric.max_steps && metric.current_step)
                ? (metric.current_step / metric.max_steps) * 100
                : 0
            });

            // Reset timing on reconnect
            startTimeRef.current = Date.now();
            lastStepRef.current = metric.current_step ?? 0;

            addLog('info', `Reconnected to training at step ${metric.current_step || 0}`);
          } else if (metric.type === 'metrics' && metric.step !== undefined) {
            // Track recent steps for rolling average ETA
            const now = Date.now();
            recentStepsRef.current.push({ step: metric.step, time: now });

            // Keep only last 50 steps for calculation (roughly 5-10 minutes of data)
            if (recentStepsRef.current.length > 50) {
              recentStepsRef.current.shift();
            }

            // Calculate ETA using rolling average (need at least 5 data points)
            if (recentStepsRef.current.length >= 5) {
              const oldestEntry = recentStepsRef.current[0];
              const newestEntry = recentStepsRef.current[recentStepsRef.current.length - 1];

              const timeElapsed = (newestEntry.time - oldestEntry.time) / 1000; // seconds
              const stepsCompleted = newestEntry.step - oldestEntry.step;
              const stepsRemaining = maxSteps - metric.step;

              if (stepsCompleted > 0 && timeElapsed > 0) {
                const stepsPerSecond = stepsCompleted / timeElapsed;
                const secondsRemaining = stepsRemaining / stepsPerSecond;

                const hours = Math.floor(secondsRemaining / 3600);
                const minutes = Math.floor((secondsRemaining % 3600) / 60);
                const seconds = Math.floor(secondsRemaining % 60);

                setEta(`${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`);
              }
            }

            // Update training state
            updateTrainingState({
              currentStep: metric.step,
              currentLoss: metric.loss ?? null,
              currentPPL: metric.perplexity ?? null,
              currentLR: metric.lr ?? null,
              progress: maxSteps > 0 ? (metric.step / maxSteps) * 100 : 0
            });

            // Add to metrics history
            addMetric({
              step: metric.step!,
              loss: metric.loss!,
              perplexity: metric.perplexity!,
              learningRate: metric.lr!,
              timestamp: metric.timestamp
            });
          } else if (metric.type === 'eval_metrics' && metric.step !== undefined && metric.eval_loss !== undefined && metric.eval_perplexity !== undefined) {
            // Update metrics with eval data
            console.log('Received eval_metrics:', metric.step, metric.eval_loss, metric.eval_perplexity);
            updateMetricWithEval(metric.step, metric.eval_loss, metric.eval_perplexity);
          } else if (metric.type === 'log' && metric.message) {
            // Add log message
            addLog(metric.level as TrainingLog['level'], metric.message);
          } else if (metric.type === 'status') {
            // Training ended
            if (metric.status === 'completed') {
              updateTrainingState({ isTraining: false });
              // Clear saved canvas configurations
              localStorage.removeItem('training_canvas_state');
            } else if (metric.status === 'error') {
              updateTrainingState({ isTraining: false });
              addLog('error', metric.error || 'Training failed');
              // Clear saved canvas configurations
              localStorage.removeItem('training_canvas_state');
            }
          }
        },
        (error) => {
          console.error('Metrics stream error:', error);
          addLog('error', `Connection error: ${error.message}`);
        }
      );

      unsubscribeRef.current = unsubscribe;

      return () => {
        if (unsubscribeRef.current) {
          unsubscribeRef.current();
          unsubscribeRef.current = null;
        }
      };
    } else {
      // Reset timing when not training
      startTimeRef.current = null;
      lastStepRef.current = 0;
      recentStepsRef.current = [];
      setEta('--:--:--');
    }
  }, [isTraining, maxSteps, updateTrainingState, addMetric, updateMetricWithEval, addLog]);

  // Check for ongoing training session on mount
  useEffect(() => {
    const checkTrainingStatus = async () => {
      try {
        const response = await fetch('/api/training/status');
        const status = await response.json();

        // If training is active but frontend doesn't know, update state
        if (status.is_training && !isTraining) {
          console.log('Detected ongoing training session, reconnecting...');
          updateTrainingState({
            isTraining: true,
            currentStep: status.current_step || 0,
            maxSteps: status.max_steps || 0,
            currentLoss: status.current_loss,
            currentPPL: status.current_ppl,
            currentLR: status.current_lr
          });
        }

      } catch (error) {
        console.error('Failed to check training status:', error);
      }
    };

    checkTrainingStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);  // Run only on mount - updateTrainingState is stable via useCallback

  const handleStop = async () => {
    try {
      await stopTraining();
      addLog('info', 'Stop requested - training will stop after current step');
    } catch (error) {
      addLog('error', `Failed to stop training: ${(error as Error).message}`);
    }
  };

  const getLogColor = (level: TrainingLog['level']) => {
    switch (level) {
      case 'info': return 'text-blue-300';
      case 'warning': return 'text-yellow-300';
      case 'error': return 'text-red-300';
      case 'success': return 'text-green-300';
    }
  };

  const getLogIcon = (level: TrainingLog['level']) => {
    switch (level) {
      case 'info': return '•';
      case 'warning': return '⚠';
      case 'error': return '✖';
      case 'success': return '✓';
    }
  };

  return (
    <div className="flex-1 bg-slate-900 p-6 overflow-y-auto">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Training Monitor</h1>
            <p className="text-slate-400">Real-time training metrics and logs</p>
          </div>

          {/* Control Buttons */}
          <div className="flex gap-3">
            {!isTraining ? (
              <div className="px-6 py-3 bg-slate-700 text-slate-400 rounded-lg font-semibold flex items-center gap-2">
                <Play className="w-5 h-5" />
                No Training Active
              </div>
            ) : (
              <button
                onClick={handleStop}
                className="px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold flex items-center gap-2 transition-colors"
              >
                <Square className="w-5 h-5" />
                Stop Training
              </button>
            )}
          </div>
        </div>

        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-slate-800 border border-slate-700 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm">Progress</span>
              <CheckCircle className="w-5 h-5 text-green-400" />
            </div>
            <div className="text-2xl font-bold text-white mb-2">{progress.toFixed(1)}%</div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-green-500 to-emerald-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-xs text-slate-400 mt-2">{currentStep} / {maxSteps} steps</div>
          </motion.div>

          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1 }}
            className="bg-slate-800 border border-slate-700 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm">Loss</span>
              <TrendingDown className="w-5 h-5 text-blue-400" />
            </div>
            <div className="text-2xl font-bold text-white">{currentLoss?.toFixed(4) || '--'}</div>
            <div className="text-xs text-slate-400 mt-2">Training loss</div>
          </motion.div>

          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="bg-slate-800 border border-slate-700 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm">Perplexity</span>
              <Zap className="w-5 h-5 text-purple-400" />
            </div>
            <div className="text-2xl font-bold text-white">{currentPPL?.toFixed(2) || '--'}</div>
            <div className="text-xs text-slate-400 mt-2">Lower is better</div>
          </motion.div>

          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="bg-slate-800 border border-slate-700 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm">Learning Rate</span>
              <TrendingUp className="w-5 h-5 text-yellow-400" />
            </div>
            <div className="text-2xl font-bold text-white">
              {currentLR ? currentLR.toExponential(2) : '--'}
            </div>
            <div className="text-xs text-slate-400 mt-2">Current LR</div>
          </motion.div>

          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="bg-slate-800 border border-slate-700 rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-sm">ETA</span>
              <Clock className="w-5 h-5 text-orange-400" />
            </div>
            <div className="text-2xl font-bold text-white">{eta}</div>
            <div className="text-xs text-slate-400 mt-2">Estimated time</div>
          </motion.div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Loss Chart */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-blue-400" />
              Training Loss
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="step"
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                />
                <Legend wrapperStyle={{ color: '#94a3b8' }} />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  name="Train Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Perplexity Chart */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-purple-400" />
              Training Perplexity
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="step"
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                />
                <Legend wrapperStyle={{ color: '#94a3b8' }} />
                <Line
                  type="monotone"
                  dataKey="perplexity"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={false}
                  name="Train PPL"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Eval Loss Chart */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-green-400" />
              Eval Loss
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={metrics.filter(m => m.evalLoss !== undefined)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="step"
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                />
                <Legend wrapperStyle={{ color: '#94a3b8' }} />
                <Line
                  type="monotone"
                  dataKey="evalLoss"
                  stroke="#22c55e"
                  strokeWidth={2}
                  dot={true}
                  name="Eval Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Eval Perplexity Chart */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-orange-400" />
              Eval Perplexity
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={metrics.filter(m => m.evalPerplexity !== undefined)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="step"
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #475569',
                    borderRadius: '0.5rem',
                    color: '#fff'
                  }}
                />
                <Legend wrapperStyle={{ color: '#94a3b8' }} />
                <Line
                  type="monotone"
                  dataKey="evalPerplexity"
                  stroke="#f97316"
                  strokeWidth={2}
                  dot={true}
                  name="Eval PPL"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Training Logs */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <Terminal className="w-5 h-5 text-green-400" />
            Training Logs
          </h3>
          <div className="bg-slate-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
            {logs.map((log, idx) => (
              <div key={idx} className="mb-1 flex gap-2">
                <span className="text-slate-500 text-xs">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className={getLogColor(log.level)}>
                  {getLogIcon(log.level)}
                </span>
                <span className="text-slate-300">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingMonitor;
