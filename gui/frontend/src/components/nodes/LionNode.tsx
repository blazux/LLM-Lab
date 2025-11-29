import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface LionNodeProps {
  data: {
    label: string;
    lr?: number;
    weight_decay?: number;
    beta1?: number;
    beta2?: number;
  };
}

export default memo(({ data }: LionNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-amber-500 to-amber-700 border-2 border-amber-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-amber-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">🦁</div>
        <div>
          <div className="text-white font-bold text-sm">Lion</div>
          <div className="text-amber-200 text-xs mt-1">
            LR: {data.lr || 0.0001}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-amber-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
