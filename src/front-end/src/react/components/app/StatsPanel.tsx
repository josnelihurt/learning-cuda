import React, { useEffect, useMemo, useState } from 'react';
import { grpcConnectionService } from '../../../infrastructure/connection/grpc-connection-service';
import { webrtcService } from '../../../infrastructure/connection/webrtc-service';
import './StatsPanel.css';

type StatsPanelProps = {
  fps: string;
  time: string;
  frames: number;
  cameraStatus: string;
  cameraStatusType: 'success' | 'error' | 'warning' | 'inactive';
  transportService?: {
    getConnectionStatus: () => {
      state: 'connected' | 'disconnected' | 'connecting' | 'error';
      lastRequest: string | null;
      lastRequestTime: Date | null;
    };
  } | null;
};

type ConnectionState = 'connected' | 'disconnected' | 'connecting' | 'error';

type ConnectionSnapshot = {
  label: 'Frames' | 'gRPC' | 'WebRTC';
  state: ConnectionState;
  lastRequest: string | null;
  lastRequestTime: Date | null;
};

function normalizeState(state: string): ConnectionState {
  if (state === 'connected' || state === 'disconnected' || state === 'connecting' || state === 'error') {
    return state;
  }
  return 'disconnected';
}

function formatElapsedTime(time: Date | null): string {
  if (!time) {
    return 'N/A';
  }
  const diffMs = Date.now() - time.getTime();
  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h`;
}

export function StatsPanel({
  fps,
  time,
  frames,
  cameraStatus,
  cameraStatusType,
  transportService = null,
}: StatsPanelProps) {
  const [connections, setConnections] = useState<ConnectionSnapshot[]>([
    { label: 'Frames', state: 'disconnected', lastRequest: null, lastRequestTime: null },
    { label: 'gRPC', state: 'disconnected', lastRequest: null, lastRequestTime: null },
    { label: 'WebRTC', state: 'disconnected', lastRequest: null, lastRequestTime: null },
  ]);

  useEffect(() => {
    const updateConnections = () => {
      const transportStatus = transportService?.getConnectionStatus() ?? {
        state: 'disconnected' as const,
        lastRequest: null,
        lastRequestTime: null,
      };
      const grpcStatus = grpcConnectionService.getConnectionStatus();
      const webRtcStatus = webrtcService.getConnectionStatus();

      setConnections([
        {
          label: 'Frames',
          state: normalizeState(transportStatus.state),
          lastRequest: transportStatus.lastRequest,
          lastRequestTime: transportStatus.lastRequestTime,
        },
        {
          label: 'gRPC',
          state: normalizeState(grpcStatus.state),
          lastRequest: grpcStatus.lastRequest,
          lastRequestTime: grpcStatus.lastRequestTime,
        },
        {
          label: 'WebRTC',
          state: normalizeState(webRtcStatus.state),
          lastRequest: webRtcStatus.lastRequest,
          lastRequestTime: webRtcStatus.lastRequestTime,
        },
      ]);
    };

    updateConnections();
    const intervalId = window.setInterval(updateConnections, 2000);
    return () => {
      clearInterval(intervalId);
    };
  }, [transportService]);

  const cameraStatusClassName = useMemo(
    () => `camera-status-${cameraStatusType}`,
    [cameraStatusType]
  );

  return (
    <div
      data-testid="react-stats-panel"
      className="react-stats-panel"
    >
      <div className="react-stats-left">
        <strong>FPS: {fps}</strong>
        <strong>Time: {time}</strong>
        <strong>Frames: {frames}</strong>
        <strong className={cameraStatusClassName}>{cameraStatus}</strong>
      </div>
      <div className="react-connections-section">
        {connections.map((connection) => (
          <div
            key={connection.label}
            className="react-connection-card"
            data-testid={`react-connection-${connection.label.toLowerCase()}`}
          >
            <div className="react-connection-header">
              <span className="react-connection-label">{connection.label}</span>
              <span className={`react-connection-indicator ${connection.state}`} />
            </div>
            <div className="react-connection-detail">
              Req: {connection.lastRequest ?? 'N/A'}
            </div>
            <div className="react-connection-time">{formatElapsedTime(connection.lastRequestTime)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
