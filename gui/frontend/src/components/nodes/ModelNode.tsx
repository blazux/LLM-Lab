import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

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
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-indigo-500 to-indigo-700 border-2 border-indigo-400 min-w-[240px]"
      style={{ cursor: 'grab' }}
    >
      {/* Left handle for datasets */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-indigo-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-4xl">🧠</div>
        <div className="flex-1">
          <div className="text-white font-bold text-sm">Model</div>
          <div className="text-indigo-200 text-xs mt-1">
            {configSource === 'current' ? 'Use current design' : 'From config file'}
          </div>
          {resumeTraining && (
            <div className="text-indigo-200 text-xs">
              Resume: {data.checkpoint_path ? '✓' : '✗'}
            </div>
          )}
        </div>
      </div>
      {/* Top handle for output */}
      <Handle
        type="source"
        position={Position.Top}
        className="!bg-indigo-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
