import { ConnectionStatus, ConnectionState } from './ConnectionStatus';

export type ConnectionType = 'websocket' | 'grpc' | 'webrtc';

export class ConnectionInfo {
  private constructor(
    public readonly type: ConnectionType,
    public readonly status: ConnectionStatus,
    public readonly label: string
  ) {
    if (!type) {
      throw new Error('Connection type cannot be empty');
    }
    if (!label) {
      throw new Error('Connection label cannot be empty');
    }
  }

  static create(
    type: ConnectionType,
    status: ConnectionStatus,
    label: string
  ): ConnectionInfo {
    return new ConnectionInfo(type, status, label);
  }

  static websocket(status: ConnectionStatus): ConnectionInfo {
    return new ConnectionInfo('websocket', status, 'ws');
  }

  static grpc(status: ConnectionStatus): ConnectionInfo {
    return new ConnectionInfo('grpc', status, 'gRPC');
  }

  static webrtc(status: ConnectionStatus): ConnectionInfo {
    return new ConnectionInfo('webrtc', status, 'WebRTC');
  }

  updateStatus(newStatus: ConnectionStatus): ConnectionInfo {
    return new ConnectionInfo(this.type, newStatus, this.label);
  }

  getStateDisplay(): string {
    const stateMap: Record<ConnectionState, string> = {
      connected: 'Connected',
      disconnected: 'Disconnected',
      connecting: 'Connecting...',
      error: 'Error',
    };
    return stateMap[this.status.state] || this.status.state;
  }

  getStateColor(): string {
    const colorMap: Record<ConnectionState, string> = {
      connected: '#66ff66',
      disconnected: '#ff6666',
      connecting: '#ffaa00',
      error: '#ff6666',
    };
    return colorMap[this.status.state] || '#b0b0b0';
  }

  equals(other: ConnectionInfo): boolean {
    return (
      this.type === other.type &&
      this.status.equals(other.status) &&
      this.label === other.label
    );
  }
}

