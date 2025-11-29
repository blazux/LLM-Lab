import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface MuonNodeProps {
  data: {
    label: string;
    lr?: number;
    weight_decay?: number;
    momentum?: number;
    nesterov?: boolean;
  };
}

export default memo(({ data }: MuonNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-orange-500 to-orange-700 border-2 border-orange-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-orange-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">🚀</div>
        <div>
          <div className="text-white font-bold text-sm">Muon</div>
          <div className="text-orange-200 text-xs mt-1">
            LR: {data.lr || 0.05}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-orange-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
