/**
 * React WebRTC Service
 * TDD RED phase - Minimal stub that will fail tests
 */

import { WebRTCSession } from '../../../domain/value-objects/WebRTCSession';
import { SignalingMessage } from '../../../gen/webrtc_signal_pb';

export class ReactWebRTCService {
  private initialized: boolean = false;
  private initPromise: Promise<void> | null = null;
  private signalingWebSockets: Map<string, WebSocket> = new Map();
  private sessions: Map<string, WebRTCSession> = new Map();
  private heartbeatIntervals: Map<string, number> = new Map();
  private peerConnections: Map<string, RTCPeerConnection> = new Map();
  private dataChannels: Map<string, RTCDataChannel> = new Map();
  private connectionState: 'connecting' | 'connected' | 'disconnected' | 'failed' = 'disconnected';
  private lastRequest: string | null = null;
  private lastRequestTime: Date | null = null;

  async initialize(): Promise<void> {
    // Not implemented yet
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  isSupported(): boolean {
    return (
      typeof RTCPeerConnection !== 'undefined' &&
      typeof RTCDataChannel !== 'undefined' &&
      typeof navigator?.mediaDevices?.getUserMedia !== 'undefined'
    );
  }

  createPeerConnection(config?: RTCConfiguration): RTCPeerConnection | null {
    // Not implemented yet
    return null;
  }

  createDataChannel(peerConnection: RTCPeerConnection, label: string): RTCDataChannel | null {
    // Not implemented yet
    return null;
  }

  async createSession(sourceId: string): Promise<WebRTCSession | null> {
    // Not implemented yet
    return null;
  }

  async closeSession(sessionId: string): Promise<void> {
    // Not implemented yet
  }

  getConnectionStatus() {
    return {
      state: this.connectionState,
      lastRequest: this.lastRequest,
      lastRequestTime: this.lastRequestTime,
    };
  }
}
