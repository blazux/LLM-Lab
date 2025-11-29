import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface MQANodeProps {
  data: {
    label: string;
    n_heads?: number;
  };
}

export default memo(({ data }: MQANodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-teal-500 to-teal-700 border-2 border-teal-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-teal-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">👓</div>
        <div>
          <div className="text-white font-bold text-sm">MQA</div>
          <div className="text-teal-200 text-xs mt-1">
            Heads: {data.n_heads || 14}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-teal-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
