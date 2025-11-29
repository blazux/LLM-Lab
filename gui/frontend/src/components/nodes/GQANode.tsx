import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface GQANodeProps {
  data: {
    label: string;
    n_heads?: number;
    n_kv_heads?: number;
  };
}

export default memo(({ data }: GQANodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-700 border-2 border-emerald-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-emerald-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">👀</div>
        <div>
          <div className="text-white font-bold text-sm">GQA</div>
          <div className="text-emerald-200 text-xs mt-1">
            H: {data.n_heads || 14} / KV: {data.n_kv_heads || 2}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-emerald-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
