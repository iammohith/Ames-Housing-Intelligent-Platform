/**
 * AgentNode — Individual DAG node with CSS animations.
 */
import React from 'react';

const STATUS_ICONS = {
  IDLE: '○', STARTED: '⟳', PROGRESS: '⟳',
  SUCCESS: '✓', FAILED: '✗', WARNING: '⚠', RETRYING: '↺',
};

export default function AgentNode({ node, style }) {
  const icon = STATUS_ICONS[node.status] || '○';
  const isAnimating = style.pulse;

  return (
    <div className={`agent-node ${isAnimating ? 'pulsing' : ''}`}
         style={{
           background: style.fill, border: `2px solid ${style.stroke}`,
           borderRadius: 8, padding: '12px 16px', minWidth: 120,
           textAlign: 'center', position: 'relative',
           boxShadow: style.glow !== 'none' ? `0 0 12px ${style.glow}40` : 'none',
           transition: 'all 0.3s ease',
         }}>
      <div style={{ fontSize: 18, color: style.stroke, fontWeight: 700 }}>
        {icon} {node.label}
      </div>
      {node.lastMessage && (
        <div style={{ fontSize: 11, color: '#8B949E', marginTop: 4, fontFamily: 'JetBrains Mono' }}>
          {node.lastMessage.substring(0, 40)}
        </div>
      )}
      {node.progress > 0 && node.status === 'PROGRESS' && (
        <div style={{ marginTop: 4, height: 3, background: '#21262D', borderRadius: 2 }}>
          <div style={{ width: `${node.progress}%`, height: '100%', background: style.stroke,
                        borderRadius: 2, transition: 'width 0.5s ease' }} />
        </div>
      )}
    </div>
  );
}
