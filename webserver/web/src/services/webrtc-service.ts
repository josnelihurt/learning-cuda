import { logger } from './otel-logger';
import { telemetryService } from './telemetry-service';
import type { IWebRTCService } from '../domain/interfaces/IWebRTCService';

class WebRTCService implements IWebRTCService {
  private initialized: boolean = false;
  private initPromise: Promise<void> | null = null;

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
}

export const webrtcService = new WebRTCService();

