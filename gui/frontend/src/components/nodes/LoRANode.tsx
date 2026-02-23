import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { motion } from 'framer-motion';
import { CircleDot } from 'lucide-react';

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
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="px-4 py-3 shadow-lg rounded-lg border border-green-500/40 min-w-[180px]"
      style={{ cursor: 'grab', backgroundColor: 'rgba(34, 197, 94, 0.08)' }}
    >
      <div className="flex items-center gap-3">
        <div className="w-1 h-14 rounded-full bg-cat-config" />
        <div className="p-2 rounded bg-slate-700">
          <CircleDot className="w-4 h-4 text-slate-300" />
        </div>
        <div className="flex-1">
          <div className="text-slate-200 font-medium text-sm">LoRA</div>
          <div className="text-slate-400 text-xs mt-0.5 space-y-0.5">
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
        className="!bg-slate-400 !border-slate-500"
        style={{ width: 10, height: 10 }}
      />
    </motion.div>
  );
});
