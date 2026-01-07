import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface MoERouterNodeProps {
  data: {
    label: string;
    num_experts?: number;
    num_experts_per_token?: number;
    load_balancing_loss_weight?: number;
    router_z_loss_weight?: number;
  };
}

export default memo(({ data }: MoERouterNodeProps) => {
  const numExperts = data.num_experts || 8;
  const topK = data.num_experts_per_token || 2;
  const loadBalance = data.load_balancing_loss_weight || 0.01;

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-purple-500 to-purple-700 border-2 border-purple-400 min-w-[200px]"
      style={{ cursor: 'grab' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-purple-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">🎯</div>
        <div>
          <div className="text-white font-bold text-sm">MoE Router</div>
          <div className="text-purple-200 text-xs mt-1">
            Creates {numExperts}× experts
          </div>
          <div className="text-purple-200 text-xs">
            Top-{topK} routing
          </div>
          <div className="text-purple-200 text-xs">
            Balance: {loadBalance}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-purple-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
