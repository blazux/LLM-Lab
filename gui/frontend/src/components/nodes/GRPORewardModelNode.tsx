import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface GRPORewardModelNodeProps {
  data: {
    label: string;
    model_name?: string;
  };
}

export default memo(({ data }: GRPORewardModelNodeProps) => {
  const displayName = data.model_name || 'OpenAssistant/reward-model-deberta-v3-large-v2';
  const shortName = displayName.split('/').pop() || displayName;

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-green-500 to-green-700 border-2 border-green-400 min-w-[240px]"
      style={{ cursor: 'grab' }}
    >
      <div className="flex items-center gap-3">
        <div className="text-3xl">👥</div>
        <div className="flex-1">
          <div className="text-white font-bold text-sm">GRPO Reward Model</div>
          <div className="text-green-200 text-xs mt-1 truncate" title={displayName}>
            {shortName}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        id="top"
        className="!bg-green-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
