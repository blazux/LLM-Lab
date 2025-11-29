import { memo } from 'react';
import { EdgeProps, getBezierPath, EdgeLabelRenderer } from 'reactflow';
import { motion } from 'framer-motion';

interface LoopEdgeData {
  repeatCount?: number;
  isLoop?: boolean;
}

export default memo(({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style = {}, data }: EdgeProps<LoopEdgeData>) => {
  const repeatCount = data?.repeatCount || 24;
  const isLoop = data?.isLoop || false;

  // Create custom path for loop edges that goes around elements
  let edgePath: string;
  let labelX: number;
  let labelY: number;

  if (isLoop) {
    // Calculate the offset to go around elements (to the left side)
    const horizontalOffset = -350; // Distance to go out to the side - much wider now
    const verticalMidpoint = (sourceY + targetY) / 2;

    // Add extra space above and below to avoid nodes
    const topPadding = 30;
    const bottomPadding = 30;

    // Create a path that goes: source -> out left -> down -> back to target
    // Using cubic bezier curves to create smooth rounded corners
    edgePath = `
      M ${sourceX},${sourceY}
      L ${sourceX - 40},${sourceY}
      C ${sourceX - 80},${sourceY} ${sourceX + horizontalOffset},${sourceY - topPadding} ${sourceX + horizontalOffset},${sourceY + 50}
      L ${sourceX + horizontalOffset},${targetY - 50}
      C ${sourceX + horizontalOffset},${targetY + bottomPadding} ${targetX - 80},${targetY} ${targetX - 40},${targetY}
      L ${targetX},${targetY}
    `.trim();

    // Position label on the left side of the loop, vertically centered
    labelX = sourceX + horizontalOffset + 20;
    labelY = verticalMidpoint;
  } else {
    // Use default bezier path for non-loop edges
    [edgePath, labelX, labelY] = getBezierPath({
      sourceX,
      sourceY,
      sourcePosition,
      targetX,
      targetY,
      targetPosition,
      curvature: 0.25,
    });
  }

  return (
    <>
      <path
        id={id}
        style={{
          ...style,
          strokeWidth: isLoop ? 5 : 2,
          stroke: isLoop ? '#a78bfa' : '#64748b',
          strokeDasharray: isLoop ? '10, 6' : 'none',
          filter: isLoop ? 'drop-shadow(0 0 10px rgba(167, 139, 250, 0.8))' : 'none',
        }}
        className="react-flow__edge-path"
        d={edgePath}
        fill="none"
      />
      {isLoop && (
        <EdgeLabelRenderer>
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'all',
            }}
            className="nodrag nopan px-4 py-2.5 bg-violet-500 border-2 border-violet-300 rounded-full shadow-lg cursor-pointer flex items-center gap-2"
            transition={{
              duration: 0.2,
            }}
          >
            <span className="text-white text-sm font-bold">↺</span>
            <span className="text-white text-sm font-bold">
              Repeat ×{repeatCount}
            </span>
          </motion.div>
        </EdgeLabelRenderer>
      )}
    </>
  );
});
