import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface HyperparametersNodeProps {
  data: {
    label: string;
    batch_size?: number;
    gradient_accumulation_steps?: number;
    max_steps?: number;
    warmup_steps?: number;
    grad_clip?: number;
    eval_every?: number;
    eval_steps?: number;
  };
}

export default memo(({ data }: HyperparametersNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-green-500 to-green-700 border-2 border-green-400 min-w-[220px]"
      style={{ cursor: 'grab' }}
    >
      <div className="flex items-center gap-3">
        <div className="text-3xl">⚙️</div>
        <div className="flex-1">
          <div className="text-white font-bold text-sm">Hyperparameters</div>
          <div className="text-green-200 text-xs mt-1 space-y-0.5">
            {data.batch_size && <div>batch: {data.batch_size}</div>}
            {data.max_steps && <div>steps: {data.max_steps}</div>}
            {data.gradient_accumulation_steps && (
              <div>grad accum: {data.gradient_accumulation_steps}</div>
            )}
          </div>
        </div>
      </div>
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-green-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
