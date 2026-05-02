/**
 * LiveDAG — D3.js DAG visualization with WebSocket-driven state transitions.
 * Renders pipeline agents as animated nodes with flowing edge particles.
 */
import React, { useEffect, useState, useRef } from 'react';

const NODE_STATES = {
  IDLE:     { fill: '#1C2333', stroke: '#3D4A5C', glow: 'none',    pulse: false },
  STARTED:  { fill: '#0A2A4A', stroke: '#4D9EFF', glow: '#4D9EFF', pulse: true  },
  PROGRESS: { fill: '#0A2A4A', stroke: '#4D9EFF', glow: '#4D9EFF', pulse: true  },
  WARNING:  { fill: '#2A1F00', stroke: '#FFB800', glow: '#FFB800', pulse: true  },
  RETRYING: { fill: '#2A1500', stroke: '#FF8C00', glow: '#FF8C00', pulse: true  },
  SUCCESS:  { fill: '#002A1A', stroke: '#00FF9C', glow: '#00FF9C', pulse: false },
  FAILED:   { fill: '#2A0008', stroke: '#FF3B30', glow: '#FF3B30', pulse: false },
};

const AGENTS = [
  { id: 'ingestion_agent',     label: 'INGESTION',     x: 100, y: 50  },
  { id: 'schema_agent',        label: 'SCHEMA',        x: 250, y: 50  },
  { id: 'cleaning_agent',      label: 'CLEANING',      x: 400, y: 50  },
  { id: 'feature_agent',       label: 'FEATURES',      x: 550, y: 50  },
  { id: 'encoding_agent',      label: 'ENCODING',      x: 700, y: 50  },
  { id: 'anomaly_agent',       label: 'ANOMALY',       x: 850, y: 20  },
  { id: 'ml_agent',            label: 'ML TRAINING',   x: 850, y: 80  },
  { id: 'orchestration_agent', label: 'ORCHESTRATION', x: 1000, y: 50 },
];

const EDGES = [
  ['ingestion_agent', 'schema_agent'],
  ['schema_agent', 'cleaning_agent'],
  ['cleaning_agent', 'feature_agent'],
  ['feature_agent', 'encoding_agent'],
  ['encoding_agent', 'anomaly_agent'],
  ['encoding_agent', 'ml_agent'],
  ['anomaly_agent', 'orchestration_agent'],
  ['ml_agent', 'orchestration_agent'],
];

export default function LiveDAG({ runId, wsUrl }) {
  const [nodes, setNodes] = useState(
    AGENTS.map(a => ({ ...a, status: 'IDLE', lastMessage: '', progress: 0 }))
  );
  const [logs, setLogs] = useState([]);
  const svgRef = useRef(null);

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(`${wsUrl}/ws/pipeline/${runId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'ping') return;

      setNodes(prev => prev.map(node =>
        node.id === data.agent
          ? { ...node, status: data.status, lastMessage: data.message,
              duration: data.duration_ms, progress: data.progress_pct || 0 }
          : node
      ));

      setLogs(prev => [...prev.slice(-500), {
        timestamp: data.timestamp, agent: data.agent,
        status: data.status, message: data.message,
      }]);
    };

    ws.onerror = () => {
      const es = new EventSource(`/api/pipeline/${runId}/events`);
      es.onmessage = (e) => ws.onmessage({ data: e.data });
    };

    return () => ws.close();
  }, [runId, wsUrl]);

  const getNodeStyle = (status) => NODE_STATES[status] || NODE_STATES.IDLE;

  return (
    <div style={{ background: '#0D1117', borderRadius: 8, padding: 16 }}>
      <svg ref={svgRef} width="1100" height="120" viewBox="0 0 1100 120">
        {/* Edges */}
        {EDGES.map(([from, to], i) => {
          const fromNode = AGENTS.find(a => a.id === from);
          const toNode = AGENTS.find(a => a.id === to);
          const fromStatus = nodes.find(n => n.id === from)?.status || 'IDLE';
          const edgeColor = fromStatus === 'SUCCESS' ? '#00FF9C' :
                           fromStatus === 'PROGRESS' ? '#4D9EFF' : '#3D4A5C';
          return (
            <line key={i} x1={fromNode.x + 50} y1={fromNode.y + 15}
                  x2={toNode.x} y2={toNode.y + 15}
                  stroke={edgeColor} strokeWidth={2}
                  strokeDasharray={fromStatus === 'SUCCESS' ? 'none' : '5,5'} />
          );
        })}
        {/* Nodes */}
        {nodes.map(node => {
          const style = getNodeStyle(node.status);
          return (
            <g key={node.id}>
              <rect x={node.x} y={node.y} width={100} height={30} rx={6}
                    fill={style.fill} stroke={style.stroke} strokeWidth={2}
                    style={style.pulse ? { animation: 'pulse 2s infinite' } : {}} />
              <text x={node.x + 50} y={node.y + 19} textAnchor="middle"
                    fill={style.stroke} fontSize={10} fontFamily="DM Sans" fontWeight={600}>
                {node.label}
              </text>
            </g>
          );
        })}
      </svg>
      <style>{`@keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.6 } }`}</style>
    </div>
  );
}
