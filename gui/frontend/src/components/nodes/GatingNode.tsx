import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface GatingNodeProps {
  data: {
    label: string;
    expand_factor?: number;
  };
}

export default memo(({ data }: GatingNodeProps) => {
  const expand_factor = data.expand_factor || 2;

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-sky-500 to-sky-700 border-2 border-sky-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-sky-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">🚪</div>
        <div>
          <div className="text-white font-bold text-sm">Gating</div>
          <div className="text-sky-200 text-xs mt-1">
            Expand: {expand_factor}x
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-sky-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
