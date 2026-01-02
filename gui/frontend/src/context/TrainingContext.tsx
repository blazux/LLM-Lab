import { createContext, useContext, useState, ReactNode } from 'react';

export interface TrainingMetrics {
  step: number;
  loss: number;
  perplexity: number;
  learningRate: number;
  timestamp: number;
  evalLoss?: number;
  evalPerplexity?: number;
}

export interface TrainingLog {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
}

interface TrainingState {
  isTraining: boolean;
  isPaused: boolean;
  progress: number;
  currentStep: number;
  maxSteps: number;
  currentLoss: number | null;
  currentPPL: number | null;
  currentLR: number | null;
  metrics: TrainingMetrics[];
  logs: TrainingLog[];
}

interface TrainingContextType {
  trainingState: TrainingState;
  setTrainingState: (state: TrainingState) => void;
  updateTrainingState: (updates: Partial<TrainingState>) => void;
  addMetric: (metric: TrainingMetrics) => void;
  updateMetricWithEval: (step: number, evalLoss: number, evalPerplexity: number) => void;
  addLog: (level: TrainingLog['level'], message: string) => void;
  clearMetrics: () => void;
}

const TrainingContext = createContext<TrainingContextType | undefined>(undefined);

export const TrainingProvider = ({ children }: { children: ReactNode }) => {
  const [trainingState, setTrainingState] = useState<TrainingState>({
    isTraining: false,
    isPaused: false,
    progress: 0,
    currentStep: 0,
    maxSteps: 10000,
    currentLoss: null,
    currentPPL: null,
    currentLR: null,
    metrics: [],
    logs: [
      {
        timestamp: new Date().toISOString(),
        level: 'info',
        message: 'Training monitor ready. Configure your model and click "Start Training".'
      }
    ],
  });

  const updateTrainingState = (updates: Partial<TrainingState>) => {
    setTrainingState(prev => ({ ...prev, ...updates }));
  };

  const addMetric = (metric: TrainingMetrics) => {
    setTrainingState(prev => ({
      ...prev,
      metrics: [...prev.metrics, metric]
    }));
  };

  const updateMetricWithEval = (step: number, evalLoss: number, evalPerplexity: number) => {
    setTrainingState(prev => {
      const existingIndex = prev.metrics.findIndex(m => m.step === step);
      if (existingIndex >= 0) {
        const updated = [...prev.metrics];
        updated[existingIndex] = {
          ...updated[existingIndex],
          evalLoss,
          evalPerplexity
        };
        return { ...prev, metrics: updated };
      }
      return prev;
    });
  };

  const addLog = (level: TrainingLog['level'], message: string) => {
    setTrainingState(prev => ({
      ...prev,
      logs: [...prev.logs, {
        timestamp: new Date().toISOString(),
        level,
        message
      }].slice(-100) // Keep last 100 logs
    }));
  };

  const clearMetrics = () => {
    setTrainingState(prev => ({
      ...prev,
      metrics: [],
      logs: [
        {
          timestamp: new Date().toISOString(),
          level: 'info',
          message: 'Training monitor ready. Configure your model and click "Start Training".'
        }
      ]
    }));
  };

  return (
    <TrainingContext.Provider value={{
      trainingState,
      setTrainingState,
      updateTrainingState,
      addMetric,
      updateMetricWithEval,
      addLog,
      clearMetrics
    }}>
      {children}
    </TrainingContext.Provider>
  );
};

export const useTraining = () => {
  const context = useContext(TrainingContext);
  if (!context) {
    throw new Error('useTraining must be used within TrainingProvider');
  }
  return context;
};
