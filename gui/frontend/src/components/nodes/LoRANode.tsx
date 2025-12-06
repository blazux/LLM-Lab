import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';

interface LoRANodeProps {
  data: {
    label: string;
    enabled?: boolean;
    preset?: string;
    lora_r?: number;
    lora_alpha?: number;
    lora_dropout?: number;
    lora_target_modules?: string[];
  };
}

export default memo(({ data }: LoRANodeProps) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-6 py-4 shadow-lg rounded-xl bg-gradient-to-br from-purple-500 to-purple-700 border-2 border-purple-400 min-w-[220px]"
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
        <div className="flex-1">
          <div className="text-white font-bold text-sm">LoRA</div>
          <div className="text-purple-200 text-xs mt-1 space-y-0.5">
            <div>preset: {data.preset || 'minimal'}</div>
            <div>rank: {data.lora_r || 8}</div>
            <div>alpha: {data.lora_alpha || 16}</div>
            {data.lora_dropout !== undefined && (
              <div>dropout: {data.lora_dropout}</div>
            )}
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
