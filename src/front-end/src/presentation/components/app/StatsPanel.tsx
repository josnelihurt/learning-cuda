import { useEffect, useId, useReducer } from 'react';
import { grpcConnectionService } from '@/infrastructure/connection/grpc-connection-service';
import { webrtcService } from '@/infrastructure/connection/webrtc-service';
import styles from './StatsPanel.module.css';

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

type StatsPanelState = {
  expanded: boolean;
  connections: ConnectionSnapshot[];
};

type StatsPanelAction =
  | { type: 'SET_CONNECTIONS'; payload: ConnectionSnapshot[] }
  | { type: 'COLLAPSE' }
  | { type: 'EXPAND' };

const INITIAL_CONNECTIONS: ConnectionSnapshot[] = [
  { label: 'Frames', state: 'disconnected', lastRequest: null, lastRequestTime: null },
  { label: 'gRPC', state: 'disconnected', lastRequest: null, lastRequestTime: null },
  { label: 'WebRTC', state: 'disconnected', lastRequest: null, lastRequestTime: null },
];

const INITIAL_STATS_PANEL_STATE: StatsPanelState = {
  expanded: true,
  connections: INITIAL_CONNECTIONS,
};

function statsPanelReducer(state: StatsPanelState, action: StatsPanelAction): StatsPanelState {
  switch (action.type) {
    case 'SET_CONNECTIONS':
      return { ...state, connections: action.payload };
    case 'COLLAPSE':
      return { ...state, expanded: false };
    case 'EXPAND':
      return { ...state, expanded: true };
    default:
      return state;
  }
}

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

function indicatorClassForState(state: ConnectionState): string {
  switch (state) {
    case 'connected':
      return styles.indicatorConnected;
    case 'connecting':
      return styles.indicatorConnecting;
    case 'error':
      return styles.indicatorError;
    default:
      return styles.indicatorDisconnected;
  }
}

function cameraClassForType(cameraStatusType: StatsPanelProps['cameraStatusType']): string {
  switch (cameraStatusType) {
    case 'success':
      return styles.cameraSuccess;
    case 'error':
      return styles.cameraError;
    case 'warning':
      return styles.cameraWarning;
    default:
      return styles.cameraInactive;
  }
}

export function StatsPanel({
  fps,
  time,
  frames,
  cameraStatus,
  cameraStatusType,
  transportService = null,
}: StatsPanelProps) {
  const panelRegionId = useId();
  const [state, dispatch] = useReducer(statsPanelReducer, INITIAL_STATS_PANEL_STATE);

  useEffect(() => {
    const updateConnections = () => {
      const transportStatus = transportService?.getConnectionStatus() ?? {
        state: 'disconnected' as const,
        lastRequest: null,
        lastRequestTime: null,
      };
      const grpcStatus = grpcConnectionService.getConnectionStatus();
      const webRtcStatus = webrtcService.getConnectionStatus();

      dispatch({
        type: 'SET_CONNECTIONS',
        payload: [
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
        ],
      });
    };

    updateConnections();
    const intervalId = window.setInterval(updateConnections, 2000);
    return () => {
      clearInterval(intervalId);
    };
  }, [transportService]);

  if (state.expanded) {
    return (
      <div
        id={panelRegionId}
        data-testid="react-stats-panel"
        className={styles.panel}
        onClick={() => dispatch({ type: 'COLLAPSE' })}
      >
        <div className={styles.left}>
          <strong>FPS: {fps}</strong>
          <strong>Time: {time}</strong>
          <strong>Frames: {frames}</strong>
          <strong className={cameraClassForType(cameraStatusType)}>{cameraStatus}</strong>
        </div>
        <div className={styles.connectionsSection}>
          {state.connections.map((connection) => (
            <div
              key={connection.label}
              className={styles.connectionCard}
              data-testid={`react-connection-${connection.label.toLowerCase()}`}
            >
              <div className={styles.connectionHeader}>
                <span className={styles.connectionLabel}>{connection.label}</span>
                <span className={indicatorClassForState(connection.state)} />
              </div>
              <div className={styles.connectionDetail}>Req: {connection.lastRequest ?? 'N/A'}</div>
              <div className={styles.connectionTime}>{formatElapsedTime(connection.lastRequestTime)}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <button
      type="button"
      className={styles.peek}
      data-testid="react-stats-panel-peek"
      aria-label="Show stats panel"
      aria-expanded={false}
      onClick={() => dispatch({ type: 'EXPAND' })}
    >
      <span className={styles.peekChevron} aria-hidden />
    </button>
  );
}
