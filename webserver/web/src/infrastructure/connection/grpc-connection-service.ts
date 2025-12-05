import { ConnectionStatus } from '../../domain/value-objects';

class GrpcConnectionService {
  private connectionState: 'connected' | 'disconnected' | 'connecting' | 'error' = 'disconnected';
  private lastRequest: string | null = null;
  private lastRequestTime: Date | null = null;

  trackRequest(requestName: string): void {
    this.lastRequest = requestName;
    this.lastRequestTime = new Date();
    this.connectionState = 'connected';
  }

  setState(state: 'connected' | 'disconnected' | 'connecting' | 'error'): void {
    this.connectionState = state;
  }

  getConnectionStatus(): { state: 'connected' | 'disconnected' | 'connecting' | 'error'; lastRequest: string | null; lastRequestTime: Date | null } {
    return {
      state: this.connectionState,
      lastRequest: this.lastRequest,
      lastRequestTime: this.lastRequestTime,
    };
  }
}

export const grpcConnectionService = new GrpcConnectionService();


