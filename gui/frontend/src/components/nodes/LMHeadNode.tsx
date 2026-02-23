import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Target } from 'lucide-react';

interface LMHeadNodeProps {
  data: {
    label: string;
    tie_weights?: boolean;
    vocab_size?: number;
  };
}

export default memo(({ data }: LMHeadNodeProps) => {
  const tieWeights = data.tie_weights ?? true;

  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-indigo-500/40 min-w-[180px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(99, 102, 241, 0.08)' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
      <div className="flex items-center gap-3">
        <div className="w-1 h-10 rounded-full bg-cat-core" />
        <div className="p-2 rounded bg-slate-700">
          <Target className="w-4 h-4 text-slate-300" />
        </div>
        <div>
          <div className="text-slate-200 font-medium text-sm">{data.label}</div>
          <div className="text-slate-400 text-xs mt-0.5">
            {tieWeights ? 'Tied weights' : 'Separate weights'}
          </div>
        </div>
      </div>
    </motion.div>
  );
});
