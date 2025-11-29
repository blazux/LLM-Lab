import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface SophiaNodeProps {
  data: {
    label: string;
    lr?: number;
    weight_decay?: number;
    beta1?: number;
    beta2?: number;
    rho?: number;
  };
}

export default memo(({ data }: SophiaNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-rose-500 to-rose-700 border-2 border-rose-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-rose-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">🎓</div>
        <div>
          <div className="text-white font-bold text-sm">Sophia</div>
          <div className="text-rose-200 text-xs mt-1">
            LR: {data.lr || 0.0001}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-rose-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
