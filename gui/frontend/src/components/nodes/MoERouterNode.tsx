import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Router } from 'lucide-react';

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
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-orange-500/40 min-w-[160px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(249, 115, 22, 0.08)' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
      <div className="flex items-center gap-3">
        <div className="w-1 h-14 rounded-full bg-cat-ffn" />
        <div className="p-2 rounded bg-slate-700">
          <Router className="w-4 h-4 text-slate-300" />
        </div>
        <div>
          <div className="text-slate-200 font-medium text-sm">MoE Router</div>
          <div className="text-slate-400 text-xs mt-0.5">
            Creates {numExperts}x experts
          </div>
          <div className="text-slate-400 text-xs">
            Top-{topK} routing
          </div>
          <div className="text-slate-400 text-xs">
            Balance: {loadBalance}
          </div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
    </motion.div>
  );
});
