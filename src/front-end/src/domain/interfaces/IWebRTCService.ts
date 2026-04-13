import type { WebRTCSession } from '../value-objects/WebRTCSession';

export interface IWebRTCService {
  initialize(): Promise<void>;
  isInitialized(): boolean;
  createPeerConnection(config?: RTCConfiguration): RTCPeerConnection | null;
  createDataChannel(peerConnection: RTCPeerConnection, label: string, options?: RTCDataChannelInit): RTCDataChannel | null;
  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream>;
  isSupported(): boolean;
  setupPingChannel(sessionId: string): Promise<void>;
  createSession(sourceId: string): Promise<WebRTCSession>;
  closeSession(sessionId: string): Promise<void>;
  startHeartbeat(sessionId: string, intervalMs: number): void;
  stopHeartbeat(sessionId: string): void;
  getActiveSessions(): WebRTCSession[];
}

