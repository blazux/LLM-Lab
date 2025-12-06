import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface RLHFHyperparametersNodeProps {
  data: {
    label: string;
    batch_size?: number;
    mini_batch_size?: number;
    learning_rate?: number;
    max_steps?: number;
    // PPO-specific
    ppo_epochs?: number;
    clip_range?: number;
    gamma?: number;
    gae_lambda?: number;
    vf_coef?: number;
    // DPO-specific
    beta?: number;
    // GRPO-specific
    group_size?: number;
    grpo_temperature?: number;
  };
}

export default memo(({ data }: RLHFHyperparametersNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-amber-500 to-amber-700 border-2 border-amber-400 min-w-[220px]"
      style={{ cursor: 'grab' }}
    >
      <div className="flex items-center gap-3">
        <div className="text-3xl">⚙️</div>
        <div className="flex-1">
          <div className="text-white font-bold text-sm">RLHF Hyperparameters</div>
          <div className="text-amber-200 text-xs mt-1 space-y-0.5">
            {data.batch_size && <div>batch: {data.batch_size}</div>}
            {data.max_steps && <div>steps: {data.max_steps}</div>}
            {data.learning_rate && <div>lr: {data.learning_rate}</div>}
            {data.ppo_epochs && <div>epochs: {data.ppo_epochs}</div>}
            {data.group_size && <div>group: {data.group_size}</div>}
          </div>
        </div>
      </div>
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-amber-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
