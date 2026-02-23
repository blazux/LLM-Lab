import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';

interface AdaptiveSchedulerNodeProps {
  data: {
    label: string;
    warmup_steps?: number;
    adaptive_window?: number;
    adaptive_patience?: number;
  };
}

export default memo(({ data }: AdaptiveSchedulerNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-violet-500/40 min-w-[160px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(139, 92, 246, 0.08)' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
      <div className="flex items-center gap-3">
        <div className="w-1 h-10 rounded-full bg-cat-sched" />
        <div className="p-2 rounded bg-slate-700">
          <Activity className="w-4 h-4 text-slate-300" />
        </div>
        <div>
          <div className="text-slate-200 font-medium text-sm">Adaptive</div>
          <div className="text-slate-400 text-xs mt-0.5">
            Window: {data.adaptive_window || 10} evals
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
    </motion.div>
  );
});
