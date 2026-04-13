/**
 * React WebRTC Service
 * Provides WebRTC peer connection, data channel, WebSocket signaling, heartbeat, and session management
 * Adapted from Lit WebRTCService pattern for React compatibility
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
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = (async () => {
      if (!this.isSupported()) {
        this.initialized = false;
        return;
      }

      this.initialized = true;
    })();

    return this.initPromise;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  isSupported(): boolean {
    return (
      typeof RTCPeerConnection !== 'undefined' &&
      typeof RTCDataChannel !== 'undefined' &&
      typeof navigator !== 'undefined' &&
      typeof navigator.mediaDevices !== 'undefined' &&
      typeof navigator.mediaDevices.getUserMedia !== 'undefined'
    );
  }

  createPeerConnection(config?: RTCConfiguration): RTCPeerConnection | null {
    if (!this.isSupported()) {
      return null;
    }

    try {
      const defaultConfig: RTCConfiguration = {
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
        ],
        ...config,
      };

      const peerConnection = new RTCPeerConnection(defaultConfig);
      return peerConnection;
    } catch (error) {
      return null;
    }
  }

  createDataChannel(
    peerConnection: RTCPeerConnection,
    label: string,
    options?: RTCDataChannelInit
  ): RTCDataChannel | null {
    if (!this.isSupported()) {
      return null;
    }

    try {
      const dataChannel = peerConnection.createDataChannel(label, options);
      return dataChannel;
    } catch (error) {
      return null;
    }
  }

  async createSession(sourceId: string): Promise<WebRTCSession | null> {
    if (!this.isSupported()) {
      throw new Error('WebRTC is not supported in this browser');
    }

    await this.initialize();
    if (!this.initialized) {
      throw new Error('WebRTC service not initialized');
    }

    const session = WebRTCSession.create(sourceId);
    const sessionId = session.getId();

    let peerConnection: RTCPeerConnection | null = null;

    try {
      peerConnection = this.createPeerConnection();
      if (!peerConnection) {
        throw new Error('Failed to create peer connection');
      }

      const dataChannel = this.createDataChannel(peerConnection, 'ping-pong-channel');
      if (!dataChannel) {
        peerConnection.close();
        throw new Error('Failed to create data channel');
      }

      this.peerConnections.set(sessionId, peerConnection);
      this.dataChannels.set(sessionId, dataChannel);

      // Set up data channel handlers
      dataChannel.onopen = () => {
        this.startHeartbeat(sessionId, 5000);
      };

      dataChannel.onmessage = (event: MessageEvent) => {
        const message = event.data;
        if (message === 'pong') {
          const currentSession = this.sessions.get(sessionId);
          if (currentSession) {
            const updatedSession = currentSession.updateHeartbeat();
            this.sessions.set(sessionId, updatedSession);
            this.connectionState = 'connected';
            this.lastRequest = 'Heartbeat pong';
            this.lastRequestTime = new Date();
          }
        }
      };

      dataChannel.onerror = (error: Event) => {
        if (dataChannel.readyState === 'closed') {
          this.stopHeartbeat(sessionId);
        }
      };

      dataChannel.onclose = () => {
        this.stopHeartbeat(sessionId);
      };

      // Create WebSocket connection
      const ws = await this.createSignalingStream(sessionId);

      // Set up ICE candidate handling
      peerConnection.onicecandidate = async (event: RTCPeerConnectionIceEvent) => {
        if (event.candidate) {
          const ws = this.signalingWebSockets.get(sessionId);
          if (ws && ws.readyState === 1) { // OPEN
            try {
              const msg = new SignalingMessage({
                message: {
                  case: 'iceCandidate',
                  value: {
                    sessionId,
                    candidate: {
                      candidate: event.candidate.candidate,
                      sdpMid: event.candidate.sdpMid || '',
                      sdpMlineIndex: event.candidate.sdpMLineIndex || 0,
                    },
                  },
                },
              });
              ws.send(msg.toJsonString());
            } catch (error) {
              // Ignore send errors
            }
          }
        }
      };

      // Create and send offer
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      if (ws.readyState === 1) { // OPEN
        const msg = new SignalingMessage({
          message: {
            case: 'startSession',
            value: {
              sessionId,
              sdpOffer: offer.sdp || '',
            },
          },
        });
        ws.send(msg.toJsonString());
      }

      this.sessions.set(sessionId, session);
      return session;
    } catch (error) {
      if (peerConnection) {
        try {
          peerConnection.close();
        } catch (cleanupError) {
          // Ignore cleanup errors
        }
        this.peerConnections.delete(sessionId);
        this.dataChannels.delete(sessionId);
      }
      return null;
    }
  }

  async closeSession(sessionId: string): Promise<void> {
    this.stopHeartbeat(sessionId);

    const ws = this.signalingWebSockets.get(sessionId);
    if (ws) {
      if (ws.readyState === 1) { // OPEN
        try {
          const msg = new SignalingMessage({
            message: {
              case: 'closeSession',
              value: {
                sessionId,
              },
            },
          });
          ws.send(msg.toJsonString());
        } catch (error) {
          // Ignore send errors
        }
      }
      try {
        ws.close();
      } catch (error) {
        // Ignore close errors
      }
      this.signalingWebSockets.delete(sessionId);
    }

    const peerConnection = this.peerConnections.get(sessionId);
    if (peerConnection) {
      try {
        peerConnection.close();
      } catch (error) {
        // Ignore close errors
      }
      this.peerConnections.delete(sessionId);
    }

    this.dataChannels.delete(sessionId);
    this.sessions.delete(sessionId);
  }

  startHeartbeat(sessionId: string, intervalMs: number): void {
    if (this.heartbeatIntervals.has(sessionId)) {
      this.stopHeartbeat(sessionId);
    }

    const dataChannel = this.dataChannels.get(sessionId);
    if (!dataChannel) {
      return;
    }

    const intervalId = window.setInterval(() => {
      const currentDataChannel = this.dataChannels.get(sessionId);
      if (!currentDataChannel) {
        this.stopHeartbeat(sessionId);
        return;
      }

      const readyState = currentDataChannel.readyState;

      if (readyState === 'open') {
        try {
          currentDataChannel.send('ping');
          this.connectionState = 'connected';
          this.lastRequest = 'Heartbeat ping';
          this.lastRequestTime = new Date();
        } catch (error) {
          this.stopHeartbeat(sessionId);
        }
      } else {
        this.stopHeartbeat(sessionId);
      }
    }, intervalMs);

    this.heartbeatIntervals.set(sessionId, intervalId);
    this.connectionState = 'connected';
  }

  stopHeartbeat(sessionId: string): void {
    const intervalId = this.heartbeatIntervals.get(sessionId);
    if (intervalId !== undefined) {
      clearInterval(intervalId);
      this.heartbeatIntervals.delete(sessionId);

      if (this.heartbeatIntervals.size === 0 && this.sessions.size === 0) {
        this.connectionState = 'disconnected';
      }
    }
  }

  getConnectionStatus() {
    const hasActiveSessions = this.sessions.size > 0;
    const hasActiveHeartbeat = this.heartbeatIntervals.size > 0;

    let currentState: 'connected' | 'disconnected' | 'connecting' | 'failed' = this.connectionState;

    if (hasActiveSessions && hasActiveHeartbeat) {
      currentState = 'connected';
    } else if (hasActiveSessions && !hasActiveHeartbeat) {
      currentState = this.connectionState === 'connecting' ? 'connecting' : 'disconnected';
    } else if (!hasActiveSessions) {
      currentState = 'disconnected';
    }

    return {
      state: currentState,
      lastRequest: this.lastRequest,
      lastRequestTime: this.lastRequestTime,
    };
  }

  private createSignalingStream(sessionId: string): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/webrtc-signaling`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        this.signalingWebSockets.set(sessionId, ws);
        resolve(ws);
      };

      ws.onmessage = async (event) => {
        try {
          const data = SignalingMessage.fromJsonString(event.data);
          this.handleSignalingMessage(data, sessionId);
        } catch (error) {
          // Ignore parse errors
        }
      };

      ws.onerror = (event: Event) => {
        reject(new Error('WebSocket connection failed'));
      };

      ws.onclose = (event: CloseEvent) => {
        this.signalingWebSockets.delete(sessionId);
      };

      const timeout = setTimeout(() => {
        if (ws.readyState !== 1) { // Not OPEN
          ws.close();
          reject(new Error('WebSocket connection timeout'));
        }
      }, 10000);
    });
  }

  private async handleSignalingMessage(msg: SignalingMessage, sessionId: string): Promise<void> {
    const peerConnection = this.peerConnections.get(sessionId);
    if (!peerConnection) {
      return;
    }

    if (msg.message.case === 'startSessionResponse') {
      const resp = msg.message.value;
      if (resp.sdpAnswer) {
        try {
          await peerConnection.setRemoteDescription(
            new RTCSessionDescription({
              type: 'answer',
              sdp: resp.sdpAnswer,
            })
          );
        } catch (error) {
          // Ignore set remote description errors
        }
      }
    } else if (msg.message.case === 'iceCandidate') {
      const resp = msg.message.value;
      if (resp.candidate) {
        try {
          await peerConnection.addIceCandidate(
            new RTCIceCandidate({
              candidate: resp.candidate.candidate,
              sdpMid: resp.candidate.sdpMid || null,
              sdpMLineIndex: resp.candidate.sdpMlineIndex || null,
            })
          );
        } catch (error) {
          // Ignore add ICE candidate errors
        }
      }
    }
  }
}
