import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Users } from 'lucide-react';

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
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-pink-500/40 min-w-[200px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(236, 72, 153, 0.08)' }}
    >
      <div className="flex items-center gap-3">
        <div className="w-1 h-10 rounded-full bg-cat-algo" />
        <div className="p-2 rounded bg-slate-700">
          <Users className="w-4 h-4 text-slate-300" />
        </div>
        <div className="flex-1">
          <div className="text-slate-200 font-medium text-sm">GRPO Reward Model</div>
          <div className="text-slate-400 text-xs mt-0.5 truncate" title={displayName}>
            {shortName}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        id="top"
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
    </motion.div>
  );
});
