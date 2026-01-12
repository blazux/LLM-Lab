import { motion } from 'framer-motion';
import { TrendingDown, Zap, Activity, TrendingUp } from 'lucide-react';

interface TrainingStatusWidgetProps {
  progress: number;
  loss: number | null;
  perplexity: number | null;
  learningRate: number | null;
  onNavigate: () => void;
}

const TrainingStatusWidget = ({ progress, loss, perplexity, learningRate, onNavigate }: TrainingStatusWidgetProps) => {
  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 20, opacity: 0 }}
      className="fixed bottom-6 right-6 z-50"
    >
      <motion.div
        whileHover={{ scale: 1.02 }}
        className="bg-gradient-to-r from-orange-600 to-red-600 rounded-lg shadow-2xl border-2 border-orange-400 cursor-pointer overflow-hidden"
        onClick={onNavigate}
      >
        {/* Pulsing animation for "live" indicator */}
        <div className="absolute top-2 right-2">
          <div className="relative">
            <span className="flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </span>
          </div>
        </div>

        <div className="p-4 min-w-[300px]">
          {/* Header */}
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-5 h-5 text-white animate-pulse" />
            <span className="text-white font-bold text-sm">Training in Progress</span>
          </div>

          {/* Progress Bar */}
          <div className="mb-3">
            <div className="flex justify-between items-center mb-1">
              <span className="text-white text-xs font-semibold">Progress</span>
              <span className="text-white text-xs font-bold">{progress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-orange-900 rounded-full h-2">
              <motion.div
                className="bg-white h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-orange-700/50 rounded-lg p-2">
              <div className="flex items-center gap-1 mb-1">
                <TrendingDown className="w-3 h-3 text-orange-200" />
                <span className="text-orange-200 text-xs">Loss</span>
              </div>
              <div className="text-white font-bold text-xs">
                {loss?.toFixed(4) || '--'}
              </div>
            </div>
            <div className="bg-orange-700/50 rounded-lg p-2">
              <div className="flex items-center gap-1 mb-1">
                <Zap className="w-3 h-3 text-orange-200" />
                <span className="text-orange-200 text-xs">PPL</span>
              </div>
              <div className="text-white font-bold text-xs">
                {perplexity?.toFixed(2) || '--'}
              </div>
            </div>
            <div className="bg-orange-700/50 rounded-lg p-2">
              <div className="flex items-center gap-1 mb-1">
                <TrendingUp className="w-3 h-3 text-orange-200" />
                <span className="text-orange-200 text-xs">LR</span>
              </div>
              <div className="text-white font-bold text-xs">
                {learningRate ? learningRate.toExponential(1) : '--'}
              </div>
            </div>
          </div>

          {/* Click to view message */}
          <div className="mt-3 text-center">
            <span className="text-white text-xs opacity-80">Click to view details →</span>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default TrainingStatusWidget;
