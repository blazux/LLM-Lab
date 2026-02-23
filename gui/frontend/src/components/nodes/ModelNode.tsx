import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { Brain } from 'lucide-react';

interface ModelNodeProps {
  data: {
    label: string;
    config_source?: 'current' | 'file';
    config_path?: string;
    checkpoint_path?: string;
    resume_training?: boolean;
  };
}

export default memo(({ data }: ModelNodeProps) => {
  const configSource = data.config_source || 'current';
  const resumeTraining = data.resume_training || false;

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
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
      <div className="flex items-center gap-3">
        {/* Category indicator */}
        <div className="w-1 h-10 rounded-full bg-cat-core" />
        <div className="p-2 rounded bg-slate-700">
          <Brain className="w-5 h-5 text-slate-300" />
        </div>
        <div className="flex-1">
          <div className="text-slate-200 font-medium text-sm">Model</div>
          <div className="text-slate-400 text-xs mt-0.5">
            {configSource === 'current' ? 'Current design' : 'From config'}
          </div>
          {resumeTraining && (
            <div className="text-slate-400 text-xs">
              Resume: {data.checkpoint_path ? 'Yes' : 'No'}
            </div>
          )}
        </div>
      </div>
      {/* Top handle for output */}
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
    </motion.div>
  );
});
