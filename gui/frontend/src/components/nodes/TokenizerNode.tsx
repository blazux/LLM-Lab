import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Type } from 'lucide-react';

interface TokenizerNodeProps {
  data: {
    label: string;
    tokenizer_name?: string;
  };
}

export default memo(({ data }: TokenizerNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-indigo-500/40 min-w-[180px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(99, 102, 241, 0.08)' }}
    >
      <div className="flex items-center gap-3">
        <div className="w-1 h-10 rounded-full bg-cat-core" />
        <div className="p-2 rounded bg-slate-700">
          <Type className="w-4 h-4 text-slate-300" />
        </div>
        <div>
          <div className="text-slate-200 font-medium text-sm">{data.label}</div>
          <div className="text-slate-400 text-xs mt-0.5">
            {data.tokenizer_name || 'Not configured'}
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
