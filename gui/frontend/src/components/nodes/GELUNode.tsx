import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface GELUNodeProps {
  data: {
    label: string;
    d_ff?: number;
  };
}

export default memo(({ data }: GELUNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-red-500 to-red-700 border-2 border-red-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-red-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">💫</div>
        <div>
          <div className="text-white font-bold text-sm">GELU FFN</div>
          <div className="text-red-200 text-xs mt-1">
            d_ff: {data.d_ff || 3584}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-red-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
