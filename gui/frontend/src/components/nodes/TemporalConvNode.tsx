import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface TemporalConvNodeProps {
  data: {
    label: string;
    conv_kernel_size?: number;
  };
}

export default memo(({ data }: TemporalConvNodeProps) => {
  const conv_kernel_size = data.conv_kernel_size || 4;

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
        <div className="text-3xl">⏱️</div>
        <div>
          <div className="text-white font-bold text-sm">Temporal Conv</div>
          <div className="text-teal-200 text-xs mt-1">
            Kernel: {conv_kernel_size}
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
