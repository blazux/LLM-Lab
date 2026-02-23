import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Eye } from 'lucide-react';

interface MHANodeProps {
  data: {
    label: string;
    n_heads?: number;
  };
}

export default memo(({ data }: MHANodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-teal-500/40 min-w-[160px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(20, 184, 166, 0.08)' }}
    >
      <Handle
        type="target"
        position={Position.Bottom}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
      <div className="flex items-center gap-3">
        <div className="w-1 h-10 rounded-full bg-cat-attention" />
        <div className="p-2 rounded bg-slate-700">
          <Eye className="w-4 h-4 text-slate-300" />
        </div>
        <div>
          <div className="text-slate-200 font-medium text-sm">MHA</div>
          <div className="text-slate-400 text-xs mt-0.5">
            Heads: {data.n_heads || 14}
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
