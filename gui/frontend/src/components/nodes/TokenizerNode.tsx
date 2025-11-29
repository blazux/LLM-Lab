import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface TokenizerNodeProps {
  data: {
    label: string;
    tokenizer_name?: string;
  };
}

export default memo(({ data }: TokenizerNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-purple-500 to-purple-700 border-2 border-purple-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <div className="flex items-center gap-3">
        <div className="text-3xl">🔤</div>
        <div>
          <div className="text-white font-bold text-sm">{data.label}</div>
          <div className="text-purple-200 text-xs mt-1">
            {data.tokenizer_name || 'Not configured'}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-purple-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
