/**
 * EdgeAnimator — SVG flowing data particles along DAG edges.
 */
import React from 'react';

export default function EdgeAnimator({ fromX, fromY, toX, toY, active, completed }) {
  const color = completed ? '#00FF9C' : active ? '#4D9EFF' : '#3D4A5C';
  const dashArray = completed ? 'none' : '6,4';

  return (
    <g>
      <line x1={fromX} y1={fromY} x2={toX} y2={toY}
            stroke={color} strokeWidth={2} strokeDasharray={dashArray}>
        {active && !completed && (
          <animate attributeName="stroke-dashoffset" from="20" to="0"
                   dur="1s" repeatCount="indefinite" />
        )}
      </line>
      {active && !completed && (
        <circle r={3} fill={color}>
          <animateMotion dur="2s" repeatCount="indefinite">
            <mpath href={`#edge-${fromX}-${toX}`} />
          </animateMotion>
        </circle>
      )}
    </g>
  );
}
