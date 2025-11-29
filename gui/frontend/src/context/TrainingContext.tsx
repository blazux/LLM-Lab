import { createContext, useContext, useState, ReactNode } from 'react';

interface TrainingState {
  isTraining: boolean;
  isPaused: boolean;
  progress: number;
  currentStep: number;
  maxSteps: number;
  currentLoss: number | null;
  currentPPL: number | null;
  currentLR: number | null;
}

interface TrainingContextType {
  trainingState: TrainingState;
  setTrainingState: (state: TrainingState) => void;
  updateTrainingState: (updates: Partial<TrainingState>) => void;
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
  });

  const updateTrainingState = (updates: Partial<TrainingState>) => {
    setTrainingState(prev => ({ ...prev, ...updates }));
  };

  return (
    <TrainingContext.Provider value={{ trainingState, setTrainingState, updateTrainingState }}>
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
