import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Database } from 'lucide-react';

interface DatasetNodeProps {
  data: {
    label: string;
    name?: string;  // Legacy field (for backwards compatibility)
    dataset_name?: string;  // Standard field
    subset?: string;
    split?: string;
    weight?: number;
  };
}

export default memo(({ data }: DatasetNodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-blue-500/40 min-w-[200px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(59, 130, 246, 0.08)' }}
    >
      <div className="flex items-center gap-3">
        <div className="w-1 h-10 rounded-full bg-cat-data" />
        <div className="p-2 rounded bg-slate-700">
          <Database className="w-4 h-4 text-slate-300" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-slate-200 font-medium text-sm">{data.label}</div>
          {(data.dataset_name || data.name) && (
            <div className="text-slate-400 text-xs mt-0.5 truncate" title={data.dataset_name || data.name}>
              {data.dataset_name || data.name}
            </div>
          )}
          {data.subset && (
            <div className="text-slate-500 text-xs">
              subset: {data.subset}
            </div>
          )}
        </div>
      </div>
      {/* Right handle connects to Data node */}
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
    </motion.div>
  );
});
