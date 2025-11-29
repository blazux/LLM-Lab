import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface ConstantSchedulerNodeProps {
  data: {
    label: string;
    warmup_steps?: number;
  };
}

export default memo(({ data: _data }: ConstantSchedulerNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-slate-500 to-slate-700 border-2 border-slate-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-slate-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">➖</div>
        <div>
          <div className="text-white font-bold text-sm">Constant</div>
          <div className="text-slate-200 text-xs mt-1">
            No decay
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-slate-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
