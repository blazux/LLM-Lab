import { motion, AnimatePresence } from 'framer-motion';
import { X, Repeat } from 'lucide-react';
import { Edge } from 'reactflow';

interface EdgeConfigPanelProps {
  edge: Edge | null;
  onClose: () => void;
  onUpdate: (edgeId: string, data: any) => void;
}

const EdgeConfigPanel = ({ edge, onClose, onUpdate }: EdgeConfigPanelProps) => {
  if (!edge || !edge.data?.isLoop) return null;

  const handleChange = (value: number) => {
    onUpdate(edge.id, {
      ...edge.data,
      repeatCount: value,
    });
  };

  const repeatCount = edge.data?.repeatCount || 24;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: 100, opacity: 0 }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="absolute bottom-4 left-1/2 -translate-x-1/2 w-96 bg-slate-800 border border-violet-500 rounded-xl shadow-2xl overflow-hidden z-20"
        style={{
          boxShadow: '0 0 40px rgba(167, 139, 250, 0.4)',
        }}
      >
        {/* Header */}
        <div className="bg-gradient-to-r from-violet-600 to-violet-500 p-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Repeat className="w-5 h-5 text-white" />
            <div>
              <h2 className="text-white font-bold text-lg">Loop Configuration</h2>
              <p className="text-violet-100 text-xs">Configure layer repetition</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-violet-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-white" />
          </button>
        </div>

        {/* Configuration */}
        <div className="p-6">
          <div className="space-y-4">
            {/* Repeat Count Slider */}
            <div>
              <label className="flex items-center justify-between text-sm font-medium text-slate-300 mb-3">
                <span>Number of Repetitions</span>
                <span className="text-violet-400 font-bold text-2xl">×{repeatCount}</span>
              </label>

              <input
                type="range"
                min="1"
                max="96"
                value={repeatCount}
                onChange={(e) => handleChange(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider-violet"
              />

              <div className="flex justify-between text-xs text-slate-500 mt-2">
                <span>1 layer</span>
                <span>96 layers</span>
              </div>
            </div>

            {/* Quick Presets */}
            <div>
              <label className="text-sm font-medium text-slate-300 mb-2 block">
                Quick Presets
              </label>
              <div className="grid grid-cols-4 gap-2">
                {[6, 12, 24, 48].map((preset) => (
                  <button
                    key={preset}
                    onClick={() => handleChange(preset)}
                    className={`
                      px-3 py-2 rounded-lg text-sm font-semibold transition-all
                      ${repeatCount === preset
                        ? 'bg-violet-500 text-white shadow-lg'
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }
                    `}
                  >
                    ×{preset}
                  </button>
                ))}
              </div>
            </div>

            {/* Info */}
            <div className="bg-slate-900 rounded-lg p-3 border border-slate-700">
              <div className="text-xs text-slate-400">
                <div className="flex items-start gap-2">
                  <span>💡</span>
                  <div>
                    <p className="font-semibold text-slate-300 mb-1">How it works:</p>
                    <p>This loop repeats the connected blocks {repeatCount} times. Each iteration processes the output from the previous one, creating a deep transformer with {repeatCount} layers.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default EdgeConfigPanel;
