import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Layers } from 'lucide-react';

interface BaseModelNodeProps {
  data: {
    label: string;
    checkpoint_path?: string;
    dropout?: number;
  };
}

export default memo(({ data }: BaseModelNodeProps) => {
  const displayPath = data.checkpoint_path || 'checkpoints/best_model.pt';
  const fileName = displayPath.split('/').pop() || displayPath;

  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-indigo-500/40 min-w-[200px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(99, 102, 241, 0.08)' }}
    >
      {/* Left handle for datasets */}
      <Handle
        type="target"
        position={Position.Left}
        id="left"
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
      {/* Bottom handle for reward/reference models */}
      <Handle
        type="target"
        position={Position.Bottom}
        id="bottom"
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
      <div className="flex items-center gap-3">
        <div className="w-1 h-10 rounded-full bg-cat-core" />
        <div className="p-2 rounded bg-slate-700">
          <Layers className="w-4 h-4 text-slate-300" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-slate-200 font-medium text-sm">Base Model</div>
          <div className="text-slate-400 text-xs mt-0.5 truncate" title={displayPath}>
            {fileName}
          </div>
        </div>
      </div>
      {/* Top handle connects to optimizer/LoRA chain */}
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
