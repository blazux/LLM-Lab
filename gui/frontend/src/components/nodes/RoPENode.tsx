import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface RoPENodeProps {
  data: {
    label: string;
    max_seq_len?: number;
  };
}

export default memo(({ data }: RoPENodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 border-2 border-blue-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-blue-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">📍</div>
        <div>
          <div className="text-white font-bold text-sm">RoPE</div>
          <div className="text-blue-200 text-xs mt-1">
            Max Length: {data.max_seq_len || 1024}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-blue-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
