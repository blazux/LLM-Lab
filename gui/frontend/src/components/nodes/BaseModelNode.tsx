import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

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
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 border-2 border-blue-400 min-w-[240px]"
      style={{ cursor: 'grab' }}
    >
      {/* Left handle for datasets */}
      <Handle
        type="target"
        position={Position.Left}
        id="left"
        className="!bg-blue-300"
        style={{ width: 12, height: 12 }}
      />
      {/* Bottom handle for reward/reference models */}
      <Handle
        type="target"
        position={Position.Bottom}
        id="bottom"
        className="!bg-blue-300"
        style={{ width: 12, height: 12 }}
      />
      <div className="flex items-center gap-3">
        <div className="text-3xl">🏗️</div>
        <div className="flex-1">
          <div className="text-white font-bold text-sm">Base Model</div>
          <div className="text-blue-200 text-xs mt-1 truncate" title={displayPath}>
            {fileName}
          </div>
          {data.dropout !== undefined && (
            <div className="text-blue-200 text-xs mt-0.5">
              Dropout: {data.dropout}
            </div>
          )}
        </div>
      </div>
      {/* Top handle connects to optimizer/LoRA chain */}
      <Handle
        type="source"
        position={Position.Top}
        id="top"
        className="!bg-blue-300"
        style={{ width: 12, height: 12 }}
      />
    </motion.div>
  );
});
