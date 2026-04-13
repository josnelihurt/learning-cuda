export type ConnectionState = 'connected' | 'disconnected' | 'connecting' | 'error';

export class ConnectionStatus {
  private constructor(
    public readonly state: ConnectionState,
    public readonly lastRequest: string | null,
    public readonly lastRequestTime: Date | null
  ) {
    if (!state) {
      throw new Error('Connection state cannot be empty');
    }
  }

  static create(
    state: ConnectionState,
    lastRequest: string | null = null,
    lastRequestTime: Date | null = null
  ): ConnectionStatus {
    return new ConnectionStatus(state, lastRequest, lastRequestTime);
  }

  static disconnected(): ConnectionStatus {
    return new ConnectionStatus('disconnected', null, null);
  }

  static connecting(): ConnectionStatus {
    return new ConnectionStatus('connecting', null, null);
  }

  static connected(lastRequest: string | null = null, lastRequestTime: Date | null = null): ConnectionStatus {
    return new ConnectionStatus('connected', lastRequest, lastRequestTime);
  }

  static error(lastRequest: string | null = null, lastRequestTime: Date | null = null): ConnectionStatus {
    return new ConnectionStatus('error', lastRequest, lastRequestTime);
  }

  updateLastRequest(request: string): ConnectionStatus {
    return new ConnectionStatus(this.state, request, new Date());
  }

  updateState(newState: ConnectionState): ConnectionStatus {
    return new ConnectionStatus(newState, this.lastRequest, this.lastRequestTime);
  }

  isConnected(): boolean {
    return this.state === 'connected';
  }

  isDisconnected(): boolean {
    return this.state === 'disconnected';
  }

  isConnecting(): boolean {
    return this.state === 'connecting';
  }

  hasError(): boolean {
    return this.state === 'error';
  }

  getLastRequestDisplay(): string {
    if (!this.lastRequest) {
      return 'N/A';
    }
    return this.lastRequest;
  }

  getLastRequestTimeDisplay(): string {
    if (!this.lastRequestTime) {
      return 'N/A';
    }
    const now = new Date();
    const diff = now.getTime() - this.lastRequestTime.getTime();
    const seconds = Math.floor(diff / 1000);
    
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

  equals(other: ConnectionStatus): boolean {
    return (
      this.state === other.state &&
      this.lastRequest === other.lastRequest &&
      this.lastRequestTime?.getTime() === other.lastRequestTime?.getTime()
    );
  }
}

