import React from 'react';

type StatsPanelProps = {
  fps: string;
  time: string;
  frames: number;
  cameraStatus: string;
  cameraStatusType: 'success' | 'error' | 'warning' | 'inactive';
};

export function StatsPanel({
  fps,
  time,
  frames,
  cameraStatus,
  cameraStatusType,
}: StatsPanelProps) {
  return (
    <div
      data-testid="react-stats-panel"
      style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        height: '35px',
        background: '#2a2a2a',
        color: '#fff',
        borderTop: '2px solid #404040',
        display: 'flex',
        alignItems: 'center',
        padding: '2px 30px 1px',
        zIndex: 1000,
        gap: '24px',
        fontSize: '12px',
      }}
    >
      <strong>FPS: {fps}</strong>
      <strong>Time: {time}</strong>
      <strong>Frames: {frames}</strong>
      <strong className={`camera-status-${cameraStatusType}`}>{cameraStatus}</strong>
    </div>
  );
}
