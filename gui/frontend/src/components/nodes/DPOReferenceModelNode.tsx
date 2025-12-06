import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface DPOReferenceModelNodeProps {
  data: {
    label: string;
    checkpoint_path?: string;
  };
}

export default memo(({ data }: DPOReferenceModelNodeProps) => {
  const displayPath = data.checkpoint_path || '(use policy checkpoint)';
  const fileName = displayPath.includes('/') ? displayPath.split('/').pop() || displayPath : displayPath;

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 border-2 border-blue-400 min-w-[240px]"
      style={{ cursor: 'grab' }}
    >
      <div className="flex items-center gap-3">
        <div className="text-3xl">🎯</div>
        <div className="flex-1">
          <div className="text-white font-bold text-sm">DPO Reference Model</div>
          <div className="text-blue-200 text-xs mt-1 truncate" title={displayPath}>
            {fileName}
          </div>
          <div className="text-blue-300 text-xs italic">
            (optional)
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        id="top"
        className="!bg-blue-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
