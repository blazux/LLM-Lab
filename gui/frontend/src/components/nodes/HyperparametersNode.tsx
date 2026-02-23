import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Settings } from 'lucide-react';

interface HyperparametersNodeProps {
  data: {
    label: string;
    batch_size?: number;
    gradient_accumulation_steps?: number;
    max_steps?: number;
    weight_decay?: number;
    grad_clip?: number;
    eval_every?: number;
    eval_steps?: number;
  };
}

export default memo(({ data }: HyperparametersNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-green-500/40 min-w-[200px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(34, 197, 94, 0.08)' }}
    >
      <div className="flex items-center gap-3">
        <div className="w-1 h-12 rounded-full bg-cat-config" />
        <div className="p-2 rounded bg-slate-700">
          <Settings className="w-4 h-4 text-slate-300" />
        </div>
        <div className="flex-1">
          <div className="text-slate-200 font-medium text-sm">Hyperparameters</div>
          <div className="text-slate-400 text-xs mt-0.5 space-y-0.5">
            {data.batch_size && <div>batch: {data.batch_size}</div>}
            {data.max_steps && <div>steps: {data.max_steps}</div>}
          </div>
        </div>
      </div>
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
    </motion.div>
  );
});
