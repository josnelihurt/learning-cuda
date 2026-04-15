import { createPromiseClient, type PromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { logger } from '../observability/otel-logger';
import { telemetryService } from '../observability/telemetry-service';
import type {
  CreateWebRTCSessionOptions,
  IWebRTCService,
} from '../../domain/interfaces/IWebRTCService';
import { WebRTCSession } from '../../domain/value-objects/WebRTCSession';
import {
  CloseSessionRequest,
  PollEventsRequest,
  SignalingMessage,
  type StartSessionResponse,
  SendIceCandidateRequest,
} from '../../gen/webrtc_signal_pb';
import { ProcessImageRequest } from '../../gen/image_processor_service_pb';
import { WebRTCSignalingService } from '../../gen/webrtc_signal_connect';
import { tracingInterceptor } from '../grpc/tracing-interceptor';

type ConnectionState = 'connected' | 'disconnected' | 'connecting' | 'error';

const DEFAULT_POLL_TIMEOUT_MS = 25_000;
const RETRY_DELAY_MS = 1_000;

export class WebRTCService implements IWebRTCService {
  private initialized = false;
  private initPromise: Promise<void> | null = null;
  private sessions: Map<string, WebRTCSession> = new Map();
  private peerConnections: Map<string, RTCPeerConnection> = new Map();
  private dataChannels: Map<string, RTCDataChannel> = new Map();
  private remoteStreamHandlers: Map<string, (stream: MediaStream) => void> = new Map();
  private remoteStreams: Map<string, MediaStream> = new Map();
  private pollControllers: Map<string, AbortController> = new Map();
  private pollCursors: Map<string, number> = new Map();
  private connectionState: ConnectionState = 'disconnected';
  private lastRequest: string | null = null;
  private lastRequestTime: Date | null = null;
  private signalingClient: PromiseClient<typeof WebRTCSignalingService>;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      interceptors: [tracingInterceptor],
      useBinaryFormat: false,
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
      const peerConnection = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
        ...config,
      });

      logger.info('Peer connection created', {
        'webrtc.ice_servers_count': peerConnection.getConfiguration().iceServers?.length || 0,
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
    if (!this.isSupported()) {
      logger.debug(`[WebRTC:${sessionId}] WebRTC not supported, skipping ping channel setup`);
      return;
    }

    await this.initialize();
    if (!this.initialized) {
      logger.debug(`[WebRTC:${sessionId}] Service not initialized, skipping ping channel setup`);
      return;
    }

    const peerConnection = this.createPeerConnection();
    if (!peerConnection) {
      logger.debug(`[WebRTC:${sessionId}] Failed to create peer connection`);
      return;
    }

    const dataChannel = this.createDataChannel(peerConnection, sessionId);
    if (!dataChannel) {
      peerConnection.close();
      return;
    }

    this.peerConnections.set(sessionId, peerConnection);
    this.dataChannels.set(sessionId, dataChannel);
    this.configureDataChannelHandlers(sessionId, dataChannel, peerConnection);

    dataChannel.onopen = () => {
      logger.debug(`[WebRTC:${sessionId}] Data channel opened`);
      dataChannel.send('ping');
      logger.debug(`[WebRTC:${sessionId}] Sent: ping`);
    };

    dataChannel.onmessage = (event: MessageEvent) => {
      const message = event.data;
      logger.debug(`[WebRTC:${sessionId}] Received: ${message}`);
      if (message === 'pong') {
        logger.info(`[WebRTC:${sessionId}] Ping-pong successful`);
      }
    };

    try {
      await this.negotiateSession(sessionId, peerConnection);
      this.ensurePolling(sessionId);
    } catch (error) {
      logger.debug(`[WebRTC:${sessionId}] Ping channel setup failed`, {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      this.cleanupSession(sessionId);
    }
  }

  async createSession(
    sourceId: string,
    options: CreateWebRTCSessionOptions = {}
  ): Promise<WebRTCSession> {
    logger.info(`[WebRTC] createSession called for sourceId: ${sourceId}`);

    if (!this.isSupported()) {
      logger.error('[WebRTC] WebRTC is not supported in this browser');
      throw new Error('WebRTC is not supported in this browser');
    }

    await this.initialize();
    if (!this.initialized) {
      logger.error('[WebRTC] WebRTC service not initialized');
      throw new Error('WebRTC service not initialized');
    }

    const sessionMode = options.mode ?? 'frame-processing';
    const session = WebRTCSession.create(sourceId, sessionMode);
    const sessionId = session.getId();

    const peerConnection = this.createPeerConnection();
    if (!peerConnection) {
      throw new Error('Failed to create peer connection');
    }

    this.peerConnections.set(sessionId, peerConnection);
    this.sessions.set(sessionId, session);
    this.connectionState = 'connecting';
    this.configurePeerConnectionStateHandlers(sessionId, peerConnection);
    this.configureRemoteTrackHandlers(sessionId, peerConnection);

    if (options.onRemoteStream) {
      this.remoteStreamHandlers.set(sessionId, options.onRemoteStream);
    }

    if (options.localStream) {
      for (const track of options.localStream.getTracks()) {
        if (track.kind === 'video' && sessionMode === 'camera-mediatrack') {
          const transceiver = peerConnection.addTransceiver(track, {
            direction: 'sendonly',
            streams: [options.localStream],
          });
          const caps = RTCRtpSender.getCapabilities('video');
          if (caps) {
            const h264Codecs = caps.codecs.filter(
              c => c.mimeType.toLowerCase() === 'video/h264'
            );
            if (h264Codecs.length > 0) {
              transceiver.setCodecPreferences(h264Codecs);
              logger.info(`[WebRTC:${sessionId}] Forced H264 codec preference for webcam track`);
            }
          }
        } else {
          peerConnection.addTrack(track, options.localStream);
        }
      }
      logger.info(`[WebRTC:${sessionId}] Local media tracks attached`, {
        'webrtc.track_count': options.localStream.getTracks().length,
        'webrtc.session_mode': sessionMode,
      });
    }

    if (sessionMode === 'camera-mediatrack') {
      peerConnection.addTransceiver('video', { direction: 'recvonly' });
      logger.info(`[WebRTC:${sessionId}] Added recvonly transceiver for processed camera video`);
    }

    const shouldUseDataChannel = options.useDataChannel ?? true;
    if (shouldUseDataChannel) {
      const dataChannel = this.createDataChannel(peerConnection, 'ping-pong-channel');
      if (!dataChannel) {
        peerConnection.close();
        throw new Error('Failed to create data channel');
      }
      this.dataChannels.set(sessionId, dataChannel);
      this.configureDataChannelHandlers(sessionId, dataChannel, peerConnection);
    }

    try {
      await this.negotiateSession(sessionId, peerConnection);
      this.ensurePolling(sessionId);

      if (shouldUseDataChannel) {
        await this.waitForDataChannelOpen(sessionId);
      }

      logger.info(`[WebRTC:${sessionId}] Session created successfully`, {
        'session.source_id': sourceId,
        'session.mode': sessionMode,
      });

      return session;
    } catch (error) {
      logger.error(`[WebRTC:${sessionId}] Failed to create session`, {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      this.cleanupSession(sessionId);
      throw error;
    }
  }

  async closeSession(sessionId: string): Promise<void> {
    try {
      logger.info(`[WebRTC:${sessionId}] Closing signaling session`);
      this.lastRequest = 'CloseSession';
      this.lastRequestTime = new Date();
      await this.signalingClient.closeSession(new CloseSessionRequest({ sessionId }));
    } catch (error) {
      logger.debug(`[WebRTC:${sessionId}] Failed to close signaling session cleanly`, {
        'error.message': error instanceof Error ? error.message : String(error),
      });
    } finally {
      this.cleanupSession(sessionId);
      if (this.sessions.size === 0) {
        this.connectionState = 'disconnected';
      }
    }
  }

  startHeartbeat(sessionId: string, intervalMs: number): void {
    void intervalMs;

    const dataChannel = this.dataChannels.get(sessionId);
    if (!dataChannel) {
      logger.debug(`[WebRTC:${sessionId}] Cannot start heartbeat: data channel not found`);
      return;
    }

    if (dataChannel.readyState === 'open') {
      this.connectionState = 'connected';
      this.lastRequest = 'Data channel ready';
      this.lastRequestTime = new Date();
    }
  }

  stopHeartbeat(sessionId: string): void {
    if (!this.sessions.has(sessionId)) {
      return;
    }

    const dataChannel = this.dataChannels.get(sessionId);
    if (!dataChannel || dataChannel.readyState !== 'open') {
      this.connectionState = 'disconnected';
    }
  }

  getActiveSessions(): WebRTCSession[] {
    return Array.from(this.sessions.values());
  }

  getDataChannel(sessionId: string): RTCDataChannel | null {
    return this.dataChannels.get(sessionId) ?? null;
  }

  sendControlRequest(sessionId: string, request: ProcessImageRequest): void {
    const dataChannel = this.dataChannels.get(sessionId);
    if (!dataChannel || dataChannel.readyState !== 'open') {
      throw new Error(`WebRTC data channel is not open for session ${sessionId}`);
    }

    const payload = request.toBinary();
    dataChannel.send(payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength));
  }

  getConnectionStatus(): { state: ConnectionState; lastRequest: string | null; lastRequestTime: Date | null } {
    const hasOpenChannel = Array.from(this.dataChannels.values()).some((channel) => channel.readyState === 'open');
    const hasConnectingChannel = Array.from(this.dataChannels.values()).some((channel) => channel.readyState === 'connecting');

    let currentState: ConnectionState = 'disconnected';
    if (this.connectionState === 'error') {
      currentState = 'error';
    } else if (this.connectionState === 'connected') {
      currentState = 'connected';
    } else if (hasOpenChannel) {
      currentState = 'connected';
    } else if (hasConnectingChannel || this.sessions.size > 0) {
      currentState = 'connecting';
    }

    return {
      state: currentState,
      lastRequest: this.lastRequest,
      lastRequestTime: this.lastRequestTime,
    };
  }

  private async negotiateSession(sessionId: string, peerConnection: RTCPeerConnection): Promise<void> {
    logger.info(`[WebRTC:${sessionId}] Negotiating signaling session over Connect`);

    peerConnection.onicecandidate = async (event: RTCPeerConnectionIceEvent) => {
      if (!event.candidate) {
        return;
      }

      try {
        this.lastRequest = 'SendIceCandidate';
        this.lastRequestTime = new Date();
        await this.signalingClient.sendIceCandidate(new SendIceCandidateRequest({
          sessionId,
          candidate: {
            candidate: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid || '',
            sdpMlineIndex: event.candidate.sdpMLineIndex || 0,
          },
        }));
        logger.debug(`[WebRTC:${sessionId}] Sent ICE candidate`);
      } catch (error) {
        this.handleSessionError(
          sessionId,
          error instanceof Error ? error : new Error(String(error))
        );
      }
    };

    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    this.lastRequest = 'StartSession';
    this.lastRequestTime = new Date();
    const response = await this.signalingClient.startSession({
      sessionId,
      sdpOffer: offer.sdp || '',
    });

    await this.applyStartSessionResponse(sessionId, response);
  }

  private async applyStartSessionResponse(
    sessionId: string,
    response: StartSessionResponse
  ): Promise<void> {
    const peerConnection = this.peerConnections.get(sessionId);
    if (!peerConnection) {
      throw new Error(`Peer connection not found for session ${sessionId}`);
    }

    if (!response.sdpAnswer) {
      throw new Error(`Missing SDP answer for session ${sessionId}`);
    }

    await peerConnection.setRemoteDescription(
      new RTCSessionDescription({
        type: 'answer',
        sdp: response.sdpAnswer,
      })
    );

    logger.info(`[WebRTC:${sessionId}] Remote description applied from StartSession response`);
  }

  private ensurePolling(sessionId: string): void {
    if (this.pollControllers.has(sessionId)) {
      return;
    }

    const abortController = new AbortController();
    this.pollControllers.set(sessionId, abortController);
    if (!this.pollCursors.has(sessionId)) {
      this.pollCursors.set(sessionId, 0);
    }

    void this.runPollLoop(sessionId, abortController);
  }

  private async runPollLoop(sessionId: string, abortController: AbortController): Promise<void> {
    while (!abortController.signal.aborted && this.peerConnections.has(sessionId)) {
      try {
        const response = await this.signalingClient.pollEvents(new PollEventsRequest({
          sessionId,
          cursor: BigInt(this.pollCursors.get(sessionId) ?? 0),
          timeoutMs: BigInt(DEFAULT_POLL_TIMEOUT_MS),
        }), {
          signal: abortController.signal,
        });

        this.pollCursors.set(sessionId, this.toNumber(response.nextCursor));

        for (const event of response.events) {
          await this.handleSignalingMessage(event, sessionId);
        }
      } catch (error) {
        if (abortController.signal.aborted) {
          return;
        }

        logger.error(`[WebRTC:${sessionId}] PollEvents failed`, {
          'error.message': error instanceof Error ? error.message : String(error),
        });

        await this.delay(RETRY_DELAY_MS);
      }
    }
  }

  private async handleSignalingMessage(message: SignalingMessage, sessionId: string): Promise<void> {
    const peerConnection = this.peerConnections.get(sessionId);
    const caseName = message.message.case;

    if (!peerConnection && caseName !== 'closeSessionResponse') {
      logger.warn(`[WebRTC:${sessionId}] Received signaling message but peer connection not found`);
      return;
    }

    if (caseName === 'startSessionResponse') {
      await this.applyStartSessionResponse(sessionId, message.message.value);
      return;
    }

    if (caseName === 'iceCandidate') {
      const candidate = message.message.value.candidate;
      if (!candidate || !peerConnection) {
        return;
      }

      await peerConnection.addIceCandidate(new RTCIceCandidate({
        candidate: candidate.candidate,
        sdpMid: candidate.sdpMid || null,
        sdpMLineIndex: candidate.sdpMlineIndex || null,
      }));

      logger.debug(`[WebRTC:${sessionId}] Added remote ICE candidate`);
      return;
    }

    if (caseName === 'iceCandidateResponse') {
      logger.debug(`[WebRTC:${sessionId}] ICE candidate acknowledged`);
      return;
    }

    if (caseName === 'closeSessionResponse') {
      logger.info(`[WebRTC:${sessionId}] Remote signaling session closed`);
      this.cleanupSession(sessionId);
      return;
    }

    if (caseName === 'keepAlive') {
      logger.debug(`[WebRTC:${sessionId}] Keepalive received`);
    }
  }

  isDataChannelOpen(sessionId: string): boolean {
    return this.dataChannels.get(sessionId)?.readyState === 'open';
  }

  private waitForDataChannelOpen(sessionId: string, timeoutMs = 10_000): Promise<void> {
    return new Promise((resolve, reject) => {
      const dc = this.dataChannels.get(sessionId);
      if (!dc) {
        reject(new Error(`Data channel not found for session ${sessionId}`));
        return;
      }
      if (dc.readyState === 'open') {
        resolve();
        return;
      }
      if (dc.readyState === 'closing' || dc.readyState === 'closed') {
        reject(new Error(`Data channel already ${dc.readyState} for session ${sessionId}`));
        return;
      }
      const cleanup = () => {
        clearTimeout(timer);
        dc.removeEventListener('open', onOpen);
        dc.removeEventListener('close', onClose);
        dc.removeEventListener('error', onClose);
      };
      const onOpen = () => {
        cleanup();
        if (dc.readyState === 'open') {
          resolve();
        } else {
          reject(new Error(`Data channel is ${dc.readyState} after open event for session ${sessionId}`));
        }
      };
      const onClose = () => {
        cleanup();
        reject(new Error(`Data channel closed while waiting for session ${sessionId}`));
      };
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error(`Data channel open timeout for session ${sessionId}`));
      }, timeoutMs);
      dc.addEventListener('open', onOpen);
      dc.addEventListener('close', onClose);
      dc.addEventListener('error', onClose);
    });
  }

  private configureDataChannelHandlers(
    sessionId: string,
    dataChannel: RTCDataChannel,
    peerConnection: RTCPeerConnection
  ): void {
    dataChannel.binaryType = 'arraybuffer';
    dataChannel.onopen = () => {
      const debugInfo: Record<string, string | number> = {
        'data_channel.label': dataChannel.label,
        'data_channel.ready_state': dataChannel.readyState,
      };

      if (dataChannel.id !== null) {
        debugInfo['data_channel.id'] = dataChannel.id;
      }

      logger.info(`[WebRTC:${sessionId}] Data channel opened`, debugInfo);
      this.connectionState = 'connected';
      this.lastRequest = 'Data channel open';
      this.lastRequestTime = new Date();
    };

    dataChannel.onmessage = (event: MessageEvent) => {
      this.connectionState = 'connected';
      this.lastRequest = typeof event.data === 'string' ? 'Data channel message' : 'Binary frame received';
      this.lastRequestTime = new Date();
    };

    dataChannel.onerror = (error: Event) => {
      logger.error(`[WebRTC:${sessionId}] Data channel error`, {
        'error.type': error.type,
        'data_channel.label': dataChannel.label,
        'data_channel.ready_state': dataChannel.readyState,
        'peer_connection.state': peerConnection.connectionState,
        'peer_connection.ice_connection_state': peerConnection.iceConnectionState,
      });

      if (dataChannel.readyState === 'closed') {
        this.connectionState = 'disconnected';
      }
    };

    dataChannel.onclose = () => {
      logger.debug(`[WebRTC:${sessionId}] Data channel closed`, {
        'data_channel.label': dataChannel.label,
        'peer_connection.state': peerConnection.connectionState,
        'peer_connection.ice_connection_state': peerConnection.iceConnectionState,
      });
      this.connectionState = 'disconnected';
    };
  }

  private configureRemoteTrackHandlers(
    sessionId: string,
    peerConnection: RTCPeerConnection
  ): void {
    peerConnection.ontrack = (event: RTCTrackEvent) => {
      // C++ may not attach a stream to the outbound track; wrap it if needed.
      const remoteStream =
        event.streams[0] ??
        (() => {
          const s = new MediaStream();
          s.addTrack(event.track);
          return s;
        })();

      this.remoteStreams.set(sessionId, remoteStream);
      this.remoteStreamHandlers.get(sessionId)?.(remoteStream);
      this.connectionState = 'connected';
      this.lastRequest = 'Remote media track';
      this.lastRequestTime = new Date();

      logger.info(`[WebRTC:${sessionId}] Remote media track received`, {
        'webrtc.track_kind': event.track.kind,
        'webrtc.stream_id': remoteStream.id,
      });
    };
  }

  private configurePeerConnectionStateHandlers(
    sessionId: string,
    peerConnection: RTCPeerConnection
  ): void {
    peerConnection.onconnectionstatechange = () => {
      const state = peerConnection.connectionState;
      logger.debug(`[WebRTC:${sessionId}] Connection state changed`, {
        'connection_state': state,
        'ice_connection_state': peerConnection.iceConnectionState,
        'signaling_state': peerConnection.signalingState,
      });

      if (state === 'connected') {
        this.connectionState = 'connected';
      } else if (state === 'failed' || state === 'disconnected' || state === 'closed') {
        this.connectionState = state === 'failed' ? 'error' : 'disconnected';
      }
    };

    peerConnection.oniceconnectionstatechange = () => {
      const iceState = peerConnection.iceConnectionState;
      logger.debug(`[WebRTC:${sessionId}] ICE connection state changed`, {
        'ice_connection_state': iceState,
        'connection_state': peerConnection.connectionState,
      });

      if (iceState === 'failed' || iceState === 'disconnected' || iceState === 'closed') {
        this.connectionState = iceState === 'failed' ? 'error' : 'disconnected';
      }
    };
  }

  private handleSessionError(sessionId: string, error: Error): void {
    logger.error(`[WebRTC:${sessionId}] Session error`, {
      'error.message': error.message,
    });

    const peerConnection = this.peerConnections.get(sessionId);
    if (!peerConnection) {
      return;
    }

    if (
      peerConnection.iceConnectionState === 'failed' ||
      peerConnection.iceConnectionState === 'disconnected'
    ) {
      this.cleanupSession(sessionId);
    }
  }

  private cleanupSession(sessionId: string): void {
    const pollController = this.pollControllers.get(sessionId);
    if (pollController) {
      pollController.abort();
      this.pollControllers.delete(sessionId);
    }
    this.pollCursors.delete(sessionId);

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

    const dataChannel = this.dataChannels.get(sessionId);
    if (dataChannel) {
      try {
        dataChannel.close();
      } catch (error) {
        logger.debug(`[WebRTC:${sessionId}] Error closing data channel`, {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      }
      this.dataChannels.delete(sessionId);
    }

    this.sessions.delete(sessionId);
    this.remoteStreams.delete(sessionId);
    this.remoteStreamHandlers.delete(sessionId);
  }

  private toNumber(value: bigint | number | string): number {
    return typeof value === 'bigint' ? Number(value) : Number(value);
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }
}

export const webrtcService = new WebRTCService();
