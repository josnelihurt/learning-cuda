export interface IWebRTCService {
  initialize(): Promise<void>;
  isInitialized(): boolean;
  createPeerConnection(config?: RTCConfiguration): RTCPeerConnection | null;
  createDataChannel(peerConnection: RTCPeerConnection, label: string, options?: RTCDataChannelInit): RTCDataChannel | null;
  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream>;
  isSupported(): boolean;
}

