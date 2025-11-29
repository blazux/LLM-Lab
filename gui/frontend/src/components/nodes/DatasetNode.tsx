import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface DatasetNodeProps {
  data: {
    label: string;
    name?: string;
    subset?: string;
    split?: string;
    weight?: number;
  };
}

export default memo(({ data }: DatasetNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 border-2 border-blue-400 min-w-[220px]"
      style={{ cursor: 'grab' }}
    >
      <div className="flex items-center gap-3">
        <div className="text-3xl">📚</div>
        <div className="flex-1">
          <div className="text-white font-bold text-sm">{data.label}</div>
          {data.name && (
            <div className="text-blue-200 text-xs mt-1 truncate" title={data.name}>
              {data.name}
            </div>
          )}
          {data.subset && (
            <div className="text-blue-200 text-xs">
              {data.subset}
            </div>
          )}
          {data.weight && (
            <div className="text-blue-200 text-xs">
              weight: {data.weight}
            </div>
          )}
        </div>
      </div>
      {/* Right handle connects to Data node */}
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-blue-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
