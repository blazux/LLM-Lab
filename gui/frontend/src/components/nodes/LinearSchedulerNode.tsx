import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface LinearSchedulerNodeProps {
  data: {
    label: string;
    warmup_steps?: number;
  };
}

export default memo(({ data }: LinearSchedulerNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-violet-500 to-violet-700 border-2 border-violet-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-violet-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">📉</div>
        <div>
          <div className="text-white font-bold text-sm">Linear</div>
          <div className="text-violet-200 text-xs mt-1">
            Warmup: {data.warmup_steps || 1000}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-violet-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
