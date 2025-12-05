import { createPromiseClient, PromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { logger } from '../observability/otel-logger';
import { telemetryService } from '../observability/telemetry-service';
import type { IWebRTCService } from '../../domain/interfaces/IWebRTCService';
import { WebRTCSignalingService } from '../../gen/webrtc_signal_connect';
import { WebRTCSession } from '../../domain/value-objects/WebRTCSession';

class WebRTCService implements IWebRTCService {
  private initialized: boolean = false;
  private initPromise: Promise<void> | null = null;
  private signalingClient: PromiseClient<typeof WebRTCSignalingService>;
  private sessions: Map<string, WebRTCSession> = new Map();
  private heartbeatIntervals: Map<string, number> = new Map();
  private peerConnections: Map<string, RTCPeerConnection> = new Map();
  private dataChannels: Map<string, RTCDataChannel> = new Map();
  private connectionState: 'connected' | 'disconnected' | 'connecting' | 'error' = 'disconnected';
  private lastRequest: string | null = null;
  private lastRequestTime: Date | null = null;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
    });
    this.signalingClient = createPromiseClient(WebRTCSignalingService, transport);
  }

  async initialize(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = telemetryService.withSpanAsync(
      'WebRTCService.initialize',
      {
        'service.name': 'WebRTCService',
        'service.method': 'initialize',
      },
      async (span) => {
        try {
          if (!this.isSupported()) {
            span?.setAttribute('webrtc.supported', false);
            logger.warn('WebRTC is not supported in this browser');
            this.initialized = false;
            return;
          }

          span?.setAttribute('webrtc.supported', true);
          span?.addEvent('WebRTC API available');

          this.initialized = true;
          logger.info('WebRTC service initialized successfully');
        } catch (error) {
          span?.setAttribute('error', true);
          span?.setAttribute('error.message', error instanceof Error ? error.message : String(error));
          logger.error('Failed to initialize WebRTC service', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          this.initialized = false;
          throw error;
        }
      }
    );

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
      logger.error('WebRTC is not supported, cannot create peer connection');
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
      logger.info('Peer connection created', {
        'webrtc.ice_servers_count': defaultConfig.iceServers?.length || 0,
      });
      return peerConnection;
    } catch (error) {
      logger.error('Failed to create peer connection', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      return null;
    }
  }

  createDataChannel(
    peerConnection: RTCPeerConnection,
    label: string,
    options?: RTCDataChannelInit
  ): RTCDataChannel | null {
    if (!this.isSupported()) {
      logger.error('WebRTC is not supported, cannot create data channel');
      return null;
    }

    try {
      const dataChannel = peerConnection.createDataChannel(label, options);
      logger.info('Data channel created', {
        'webrtc.data_channel.label': label,
        'webrtc.data_channel.ordered': options?.ordered !== false,
      });
      return dataChannel;
    } catch (error) {
      logger.error('Failed to create data channel', {
        'error.message': error instanceof Error ? error.message : String(error),
        'webrtc.data_channel.label': label,
      });
      return null;
    }
  }

  async getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream> {
    if (!this.isSupported()) {
      throw new Error('WebRTC is not supported in this browser');
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      logger.info('User media stream obtained', {
        'webrtc.media.audio': constraints.audio !== false,
        'webrtc.media.video': constraints.video !== false,
      });
      return stream;
    } catch (error) {
      logger.error('Failed to get user media', {
        'error.message': error instanceof Error ? error.message : String(error),
        'webrtc.media.audio': constraints.audio !== false,
        'webrtc.media.video': constraints.video !== false,
      });
      throw error;
    }
  }

  async setupPingChannel(sessionId: string): Promise<void> {
    // Early validation to avoid unnecessary work
    if (!this.isSupported()) {
      logger.debug(`[WebRTC:${sessionId}] WebRTC not supported, skipping ping channel setup`);
      return;
    }

    let peerConnection: RTCPeerConnection | null = null;

    try {
      await this.initialize();
      if (!this.initialized) {
        logger.debug(`[WebRTC:${sessionId}] Service not initialized, skipping ping channel setup`);
        return;
      }

      peerConnection = this.createPeerConnection();
      if (!peerConnection) {
        logger.debug(`[WebRTC:${sessionId}] Failed to create peer connection`);
        return;
      }

      const dataChannel = this.createDataChannel(peerConnection, 'ping-pong-channel');
      if (!dataChannel) {
        logger.debug(`[WebRTC:${sessionId}] Failed to create data channel`);
        peerConnection.close();
        peerConnection = null;
        return;
      }

      // Configure data channel callbacks
      dataChannel.onopen = () => {
        console.log(`[WebRTC:${sessionId}] Data channel opened`);
        dataChannel.send('ping');
        console.log(`[WebRTC:${sessionId}] Sent: ping`);
      };

      dataChannel.onmessage = (event: MessageEvent) => {
        const message = event.data;
        console.log(`[WebRTC:${sessionId}] Received: ${message}`);
        if (message === 'pong') {
          logger.info(`[WebRTC:${sessionId}] Ping-pong successful!`);
        }
      };

      dataChannel.onerror = (error: Event) => {
        logger.error(`[WebRTC:${sessionId}] Data channel error`, {
          'error.type': error.type,
        });
      };

      dataChannel.onclose = () => {
        console.log(`[WebRTC:${sessionId}] Data channel closed`);
      };

      // Handle ICE candidates
      peerConnection.onicecandidate = async (event: RTCPeerConnectionIceEvent) => {
        if (event.candidate) {
          try {
            await this.signalingClient.sendIceCandidate({
              sessionId,
              candidate: {
                candidate: event.candidate.candidate,
                sdpMid: event.candidate.sdpMid || '',
                sdpMlineIndex: event.candidate.sdpMLineIndex || 0,
              },
            });
            console.log(`[WebRTC:${sessionId}] Sent ICE candidate: ${event.candidate.candidate}`);
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`[WebRTC:${sessionId}] Failed to send ICE candidate`, {
              'error.message': errorMessage,
            });
          }
        }
      };

      peerConnection.onconnectionstatechange = () => {
        if (peerConnection) {
          console.log(`[WebRTC:${sessionId}] Connection state: ${peerConnection.connectionState}`);
        }
      };

      // Create and send offer
      if (!peerConnection) {
        logger.debug(`[WebRTC:${sessionId}] Peer connection lost before creating offer`);
        return;
      }

      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      const startResponse = await this.signalingClient.startSession({
        sessionId,
        sdpOffer: offer.sdp || '',
      });

      if (!startResponse.sdpAnswer) {
        logger.debug(`[WebRTC:${sessionId}] No SDP answer received`);
        if (peerConnection) {
          peerConnection.close();
          peerConnection = null;
        }
        return;
      }

      if (!peerConnection) {
        logger.debug(`[WebRTC:${sessionId}] Peer connection lost before setting remote description`);
        return;
      }

      await peerConnection.setRemoteDescription(
        new RTCSessionDescription({
          type: 'answer',
          sdp: startResponse.sdpAnswer,
        })
      );

      logger.debug(`[WebRTC:${sessionId}] WebRTC connection established`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.debug(`[WebRTC:${sessionId}] Ping channel setup failed`, {
        'error.message': errorMessage,
      });
      // Cleanup resources on error
      if (peerConnection) {
        try {
          peerConnection.close();
        } catch (cleanupError) {
          // Ignore cleanup errors
        }
        peerConnection = null;
      }
    }
  }

  async createSession(sourceId: string): Promise<WebRTCSession> {
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

      dataChannel.onopen = () => {
        logger.info(`[WebRTC:${sessionId}] Data channel opened - readyState: ${dataChannel.readyState}`);
        const channelId = dataChannel.id;
        const debugInfo: Record<string, string | number | boolean> = {
          'data_channel.label': dataChannel.label,
          'data_channel.ready_state': dataChannel.readyState,
        };
        if (channelId !== null && channelId !== undefined) {
          debugInfo['data_channel.id'] = channelId;
        }
        logger.debug(`[WebRTC:${sessionId}] Data channel details`, debugInfo);
        // Start heartbeat when data channel is ready
        logger.info(`[WebRTC:${sessionId}] Starting heartbeat in 1 second...`);
        setTimeout(() => {
          this.startHeartbeat(sessionId, 5000);
        }, 1000);
      };

      dataChannel.onmessage = (event: MessageEvent) => {
        const message = event.data;
        logger.debug(`[WebRTC:${sessionId}] Received message on data channel: ${message}`);
        if (message === 'pong') {
          const currentSession = this.sessions.get(sessionId);
          if (currentSession) {
            const updatedSession = currentSession.updateHeartbeat();
            this.sessions.set(sessionId, updatedSession);
            this.connectionState = 'connected';
            this.lastRequest = 'Heartbeat pong';
            this.lastRequestTime = new Date();
            logger.info(`[WebRTC:${sessionId}] Heartbeat pong received and updated`);
          } else {
            logger.warn(`[WebRTC:${sessionId}] Received pong but session not found`);
          }
        } else {
          logger.debug(`[WebRTC:${sessionId}] Received non-pong message: ${message}`);
        }
      };

      dataChannel.onerror = (error: Event) => {
        const errorDetails: Record<string, any> = {
          'error.type': error.type,
          'data_channel.label': dataChannel.label,
          'data_channel.ready_state': dataChannel.readyState,
          'data_channel.buffered_amount': dataChannel.bufferedAmount,
        };
        
        if (error.target instanceof RTCDataChannel) {
          errorDetails['data_channel.id'] = error.target.id;
        }
        
        if (peerConnection) {
          errorDetails['peer_connection.state'] = peerConnection.connectionState;
          errorDetails['peer_connection.ice_connection_state'] = peerConnection.iceConnectionState;
          errorDetails['peer_connection.ice_gathering_state'] = peerConnection.iceGatheringState;
          errorDetails['peer_connection.signaling_state'] = peerConnection.signalingState;
        }
        
        logger.error(`[WebRTC:${sessionId}] Data channel error`, errorDetails);
        
        // Log additional diagnostic information
        if (peerConnection) {
          logger.debug(`[WebRTC:${sessionId}] Diagnostic info on data channel error`, {
            'data_channel.ready_state': dataChannel.readyState,
            'peer_connection.state': peerConnection.connectionState,
            'peer_connection.ice_connection_state': peerConnection.iceConnectionState,
            'peer_connection.signaling_state': peerConnection.signalingState,
            'peer_connection.ice_gathering_state': peerConnection.iceGatheringState,
          });
        }
        
        // If data channel is closed, stop heartbeat
        if (dataChannel.readyState === 'closed') {
          logger.debug(`[WebRTC:${sessionId}] Data channel is closed, stopping heartbeat`);
          this.stopHeartbeat(sessionId);
        }
      };

      dataChannel.onclose = () => {
        const closeDetails: Record<string, any> = {
          'data_channel.label': dataChannel.label,
          'data_channel.ready_state': dataChannel.readyState,
        };
        
        if (peerConnection) {
          closeDetails['peer_connection.state'] = peerConnection.connectionState;
          closeDetails['peer_connection.ice_connection_state'] = peerConnection.iceConnectionState;
        }
        
        logger.debug(`[WebRTC:${sessionId}] Data channel closed`, closeDetails);
        // Stop heartbeat when data channel closes
        this.stopHeartbeat(sessionId);
      };

      peerConnection.onicecandidate = async (event: RTCPeerConnectionIceEvent) => {
        if (event.candidate && peerConnection) {
          try {
            await this.signalingClient.sendIceCandidate({
              sessionId,
              candidate: {
                candidate: event.candidate.candidate,
                sdpMid: event.candidate.sdpMid || '',
                sdpMlineIndex: event.candidate.sdpMLineIndex || 0,
              },
            });
            logger.debug(`[WebRTC:${sessionId}] Sent ICE candidate`);
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`[WebRTC:${sessionId}] Failed to send ICE candidate`, {
              'error.message': errorMessage,
            });
          }
        }
      };


      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      const startResponse = await this.signalingClient.startSession({
        sessionId,
        sdpOffer: offer.sdp || '',
      });

      if (!startResponse.sdpAnswer) {
        peerConnection.close();
        this.peerConnections.delete(sessionId);
        this.dataChannels.delete(sessionId);
        throw new Error('No SDP answer received');
      }

      await peerConnection.setRemoteDescription(
        new RTCSessionDescription({
          type: 'answer',
          sdp: startResponse.sdpAnswer,
        })
      );

      // Check data channel state after setting remote description
      const dataChannelState = dataChannel.readyState;
      logger.debug(`[WebRTC:${sessionId}] Data channel state after setRemoteDescription: ${dataChannelState}`);

      // If data channel is already open, start heartbeat immediately
      if (dataChannelState === 'open') {
        logger.debug(`[WebRTC:${sessionId}] Data channel already open, starting heartbeat`);
        this.startHeartbeat(sessionId, 5000);
      }

      // Monitor connection state changes
      peerConnection.onconnectionstatechange = () => {
        if (peerConnection) {
          const state = peerConnection.connectionState;
          const iceState = peerConnection.iceConnectionState;
          logger.debug(`[WebRTC:${sessionId}] Connection state changed`, {
            'connection_state': state,
            'ice_connection_state': iceState,
            'signaling_state': peerConnection.signalingState,
          });
          
          // Check data channel state when connection is established
          if (state === 'connected') {
            const currentDataChannel = this.dataChannels.get(sessionId);
            if (currentDataChannel) {
              const dcState = currentDataChannel.readyState;
              logger.debug(`[WebRTC:${sessionId}] Data channel state when connection ${state}: ${dcState}`);
              
              if (dcState === 'open' && !this.heartbeatIntervals.has(sessionId)) {
                logger.debug(`[WebRTC:${sessionId}] Data channel opened, starting heartbeat`);
                this.startHeartbeat(sessionId, 5000);
              }
            }
          } else if (state === 'failed' || state === 'disconnected' || state === 'closed') {
            logger.warn(`[WebRTC:${sessionId}] Connection ${state}, stopping heartbeat`);
            this.stopHeartbeat(sessionId);
          }
        }
      };
      
      peerConnection.oniceconnectionstatechange = () => {
        if (peerConnection) {
          const iceState = peerConnection.iceConnectionState;
          logger.debug(`[WebRTC:${sessionId}] ICE connection state changed`, {
            'ice_connection_state': iceState,
            'connection_state': peerConnection.connectionState,
          });
          
          if (iceState === 'failed' || iceState === 'disconnected' || iceState === 'closed') {
            logger.warn(`[WebRTC:${sessionId}] ICE connection ${iceState}, stopping heartbeat`);
            this.stopHeartbeat(sessionId);
          }
        }
      };

      this.sessions.set(sessionId, session);
      logger.info(`[WebRTC:${sessionId}] Session created successfully`, {
        'session.source_id': sourceId,
      });

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
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error(`[WebRTC:${sessionId}] Failed to create session`, {
        'error.message': errorMessage,
      });
      throw error;
    }
  }

  async closeSession(sessionId: string): Promise<void> {
    this.stopHeartbeat(sessionId);

    const peerConnection = this.peerConnections.get(sessionId);
    if (peerConnection) {
      try {
        peerConnection.close();
      } catch (error) {
        logger.debug(`[WebRTC:${sessionId}] Error closing peer connection`, {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      }
      this.peerConnections.delete(sessionId);
    }

    this.dataChannels.delete(sessionId);
    this.sessions.delete(sessionId);

    try {
      if ('closeSession' in this.signalingClient && typeof this.signalingClient.closeSession === 'function') {
        await (this.signalingClient as any).closeSession({ sessionId });
        logger.info(`[WebRTC:${sessionId}] Session closed successfully`);
      } else {
        logger.debug(`[WebRTC:${sessionId}] CloseSession not yet available in signaling client (proto not regenerated)`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.debug(`[WebRTC:${sessionId}] Failed to close session on backend (may already be closed)`, {
        'error.message': errorMessage,
      });
    }
  }

  startHeartbeat(sessionId: string, intervalMs: number): void {
    if (this.heartbeatIntervals.has(sessionId)) {
      this.stopHeartbeat(sessionId);
    }

    const dataChannel = this.dataChannels.get(sessionId);
    if (!dataChannel) {
      logger.debug(`[WebRTC:${sessionId}] Cannot start heartbeat: data channel not found`);
      return;
    }

    logger.info(`[WebRTC:${sessionId}] Setting up heartbeat interval (${intervalMs}ms)`);
    const intervalId = window.setInterval(() => {
      const currentDataChannel = this.dataChannels.get(sessionId);
      if (!currentDataChannel) {
        logger.warn(`[WebRTC:${sessionId}] Data channel not found, stopping heartbeat`);
        this.stopHeartbeat(sessionId);
        return;
      }
      
      const readyState = currentDataChannel.readyState;
      logger.debug(`[WebRTC:${sessionId}] Heartbeat check - data channel state: ${readyState}`);
      
      if (readyState === 'open') {
        try {
          currentDataChannel.send('ping');
          this.connectionState = 'connected';
          this.lastRequest = 'Heartbeat ping';
          this.lastRequestTime = new Date();
          logger.info(`[WebRTC:${sessionId}] Heartbeat ping sent successfully`);
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);
          logger.error(`[WebRTC:${sessionId}] Failed to send heartbeat ping`, {
            'error.message': errorMessage,
            'data_channel.ready_state': readyState,
            'data_channel.buffered_amount': currentDataChannel.bufferedAmount,
          });
          this.stopHeartbeat(sessionId);
        }
      } else {
        logger.warn(`[WebRTC:${sessionId}] Data channel not open (state: ${readyState}), stopping heartbeat`);
        this.stopHeartbeat(sessionId);
      }
    }, intervalMs);

    this.heartbeatIntervals.set(sessionId, intervalId);
    this.connectionState = 'connected';
    logger.info(`[WebRTC:${sessionId}] Heartbeat started`, {
      'heartbeat.interval_ms': intervalMs,
    });
  }

  stopHeartbeat(sessionId: string): void {
    const intervalId = this.heartbeatIntervals.get(sessionId);
    if (intervalId !== undefined) {
      clearInterval(intervalId);
      this.heartbeatIntervals.delete(sessionId);
      logger.debug(`[WebRTC:${sessionId}] Heartbeat stopped`);
      
      if (this.heartbeatIntervals.size === 0 && this.sessions.size === 0) {
        this.connectionState = 'disconnected';
      }
    }
  }

  getActiveSessions(): WebRTCSession[] {
    return Array.from(this.sessions.values());
  }

  getConnectionStatus(): { state: 'connected' | 'disconnected' | 'connecting' | 'error'; lastRequest: string | null; lastRequestTime: Date | null } {
    const hasActiveSessions = this.sessions.size > 0;
    const hasActiveHeartbeat = this.heartbeatIntervals.size > 0;
    
    let currentState: 'connected' | 'disconnected' | 'connecting' | 'error' = this.connectionState;
    
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
}

export const webrtcService = new WebRTCService();

