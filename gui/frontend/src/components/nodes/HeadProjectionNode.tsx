import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface HeadProjectionNodeProps {
  data: {
    label: string;
    headdim?: number;
    ngroups?: number;
  };
}

export default memo(({ data }: HeadProjectionNodeProps) => {
  const headdim = data.headdim || 64;
  const ngroups = data.ngroups || 1;

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-indigo-500 to-indigo-700 border-2 border-indigo-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-indigo-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">🧠</div>
        <div>
          <div className="text-white font-bold text-sm">Head Projection</div>
          <div className="text-indigo-200 text-xs mt-1">
            Dim: {headdim}, Groups: {ngroups}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-indigo-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
