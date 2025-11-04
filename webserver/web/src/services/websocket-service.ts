import type { StatsPanel } from '../components/stats-panel';
import type { CameraPreview } from '../components/camera-preview';
import type { ToastContainer } from '../components/toast-container';
import {
  WebSocketFrameRequest,
  WebSocketFrameResponse,
  ProcessImageRequest,
  StartVideoPlaybackRequest,
  StopVideoPlaybackRequest,
} from '../gen/image_processor_service_pb';
import { FilterType, AcceleratorType, GrayscaleType, TraceContext, GaussianBlurParameters, BorderMode } from '../gen/common_pb';
import { streamConfigService } from './config-service';
import { telemetryService } from './telemetry-service';
import { logger } from './otel-logger';
import { context, propagation } from '@opentelemetry/api';
import type { IWebSocketService } from '../domain/interfaces/IWebSocketService';
import { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '../domain/value-objects';

type FrameResultCallback = (data: WebSocketFrameResponse) => void;

function extractBlurParams(filters: FilterData[]): GaussianBlurParameters | undefined {
  const blurFilter = filters.find(f => f.isBlur());
  if (!blurFilter) {
    return undefined;
  }

  const params = blurFilter.getParameters();
  const kernelSize = params.kernel_size !== undefined ? params.kernel_size : 5;
  const sigma = params.sigma !== undefined ? params.sigma : 1.0;
  const separable = params.separable !== undefined ? params.separable : true;
  
  let borderMode = BorderMode.REFLECT;
  if (params.border_mode !== undefined) {
    const borderModeStr = String(params.border_mode).toUpperCase();
    switch (borderModeStr) {
      case 'CLAMP':
        borderMode = BorderMode.CLAMP;
        break;
      case 'REFLECT':
        borderMode = BorderMode.REFLECT;
        break;
      case 'WRAP':
        borderMode = BorderMode.WRAP;
        break;
      default:
        borderMode = BorderMode.REFLECT;
    }
  }

  return new GaussianBlurParameters({
    kernelSize,
    sigma,
    borderMode,
    separable,
  });
}

// TODO: To be replaced by Connect-RPC bidirectional streaming client
// Target replacement: Use createPromiseClient with ImageProcessorService.streamProcessVideo
// Reference implementation: webserver/pkg/interfaces/connectrpc/handler.go StreamProcessVideo method
// Migration: Use @connectrpc/connect-web streaming API instead of native WebSocket
// Benefits: Type-safe, automatic reconnection, unified with other RPC calls, better error handling
// Keep during migration for backward compatibility
export class WebSocketService implements IWebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimeout = 3000;
  private onFrameResultCallback: FrameResultCallback | null = null;

  constructor(
    private statsManager: StatsPanel,
    private cameraManager: CameraPreview,
    private toastManager: ToastContainer
  ) {}

  connect(): void {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const endpoint = streamConfigService.getWebSocketEndpoint();
    this.ws = new WebSocket(`${protocol}//${window.location.host}${endpoint}`);

    this.ws.onopen = () => {
      this.statsManager.updateWebSocketStatus('connected', 'Connected');
      logger.info('WebSocket connected');
    };

    this.ws.onmessage = async (event) => {
      const receiveTime = performance.now();
      this.cameraManager.setProcessing(false);

      try {
        let data: WebSocketFrameResponse;
        const transportFormat = streamConfigService.getTransportFormat();

        if (transportFormat === 'binary') {
          const buffer = await event.data.arrayBuffer();
          data = WebSocketFrameResponse.fromBinary(new Uint8Array(buffer));
        } else {
          data = WebSocketFrameResponse.fromJsonString(event.data);
        }

        if (data.type === 'frame_result' || data.type === 'video_frame') {
          if (data.success) {
            const sendTime = this.cameraManager.getLastFrameTime();
            const processingTime = receiveTime - sendTime;

            this.statsManager.updateProcessingStats(processingTime);

            if (this.onFrameResultCallback) {
              this.onFrameResultCallback(data);
            }

            if (data.type === 'video_frame' && data.videoFrame) {
              logger.debug(
                `Video frame received: ${data.videoFrame.frameNumber} frame_id: ${data.videoFrame.frameId}`,
                {
                  'video.frame_number': data.videoFrame.frameNumber,
                  'video.frame_id': data.videoFrame.frameId,
                }
              );
            }
          } else {
            logger.error('Frame processing error', {
              'error.message': data.error || 'Unknown error',
            });
            this.toastManager.error('Processing Error', data.error || 'Unknown error');
            this.statsManager.updateCameraStatus('Processing failed', 'error');
          }
        }
      } catch (error) {
        logger.error('Error parsing WebSocket message', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      }
    };

    this.ws.onerror = (error) => {
      logger.error('WebSocket error', {
        'error.type': error.type,
      });
      this.toastManager.warning('WebSocket Error', 'Connection error, attempting to reconnect...');
      this.statsManager.updateWebSocketStatus('disconnected', 'Connection error');
      this.cameraManager.setProcessing(false);
    };

    this.ws.onclose = () => {
      logger.info('WebSocket closed - Reconnecting...');
      this.statsManager.updateWebSocketStatus('connecting', 'Reconnecting...');
      setTimeout(() => this.connect(), this.reconnectTimeout);
    };
  }

  sendFrame(
    base64Data: string,
    width: number,
    height: number,
    filters: string[],
    accelerator: string,
    grayscaleType: string,
    blurParams?: Record<string, any>
  ): void {
    const image = new ImageData(base64Data, width, height);
    this.sendFrameWithImageData(image, filters, accelerator, grayscaleType, blurParams);
  }

  sendFrameWithImageData(image: ImageData, filters: string[], accelerator: string, grayscaleType: string, blurParams?: Record<string, any>): void {
    const filterObjects = filters.map(f => {
      if (f === 'blur' && blurParams) {
        return new FilterData('blur', blurParams);
      }
      return new FilterData(f as any);
    });
    this.sendFrameWithValueObjects(image, filterObjects, accelerator, grayscaleType);
  }

  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: string, grayscaleType: string): void {
    if (!this.ws || this.ws.readyState !== 1) return; // WebSocket.OPEN

    this.cameraManager.setProcessing(true);

    const protoFilters = filters.map(f => f.toProtocol());

    const protoAccelerator = accelerator === 'cpu' ? AcceleratorType.CPU : AcceleratorType.CUDA;

    let protoGrayscaleType: GrayscaleType;
    switch (grayscaleType) {
      case 'bt601':
        protoGrayscaleType = GrayscaleType.BT601;
        break;
      case 'bt709':
        protoGrayscaleType = GrayscaleType.BT709;
        break;
      case 'average':
        protoGrayscaleType = GrayscaleType.AVERAGE;
        break;
      case 'lightness':
        protoGrayscaleType = GrayscaleType.LIGHTNESS;
        break;
      case 'luminosity':
        protoGrayscaleType = GrayscaleType.LUMINOSITY;
        break;
      default:
        protoGrayscaleType = GrayscaleType.BT601;
    }

    const imageDataB64 = image.getBase64().replace(/^data:image\/(png|jpeg);base64,/, '');
    const imageBytes = Uint8Array.from(atob(imageDataB64), (c) => c.charCodeAt(0));

    const blurParams = extractBlurParams(filters);

    const request = new ProcessImageRequest({
      imageData: imageBytes,
      width: image.getWidth(),
      height: image.getHeight(),
      channels: 4,
      filters: protoFilters,
      accelerator: protoAccelerator,
      grayscaleType: protoGrayscaleType,
      blurParams: blurParams,
    });

    const carrier: { [key: string]: string } = {};
    propagation.inject(context.active(), carrier);

    const traceContext = new TraceContext({
      traceparent: carrier['traceparent'] || '',
      tracestate: carrier['tracestate'] || '',
    });

    const frameRequest = new WebSocketFrameRequest({
      type: 'frame',
      request: request,
      traceContext: traceContext,
    });

    const transportFormat = streamConfigService.getTransportFormat();
    if (transportFormat === 'binary') {
      this.ws.send(frameRequest.toBinary());
    } else {
      this.ws.send(frameRequest.toJsonString());
    }

    logger.debug('Frame sent via WebSocket', {
      width: image.getWidth(),
      height: image.getHeight(),
      aspectRatio: image.getAspectRatio(),
      filterCount: filters.length,
      filterTypes: filters.map(f => f.getType()).join(','),
    });
  }

  sendFrameWithProcessingConfig(image: ImageData, filters: FilterData[], accelerator: AcceleratorConfig, grayscale: GrayscaleAlgorithm): void {
    if (!this.ws || this.ws.readyState !== 1) return; // WebSocket.OPEN

    this.cameraManager.setProcessing(true);

    const protoFilters = filters.map(f => f.toProtocol());
    const protoAccelerator = accelerator.toProtocol();
    const protoGrayscaleType = grayscale.toProtocol();

    const imageDataB64 = image.getBase64().replace(/^data:image\/(png|jpeg);base64,/, '');
    const imageBytes = Uint8Array.from(atob(imageDataB64), (c) => c.charCodeAt(0));

    const blurParams = extractBlurParams(filters);

    const request = new ProcessImageRequest({
      imageData: imageBytes,
      width: image.getWidth(),
      height: image.getHeight(),
      channels: 4,
      filters: protoFilters,
      accelerator: protoAccelerator,
      grayscaleType: protoGrayscaleType,
      blurParams: blurParams,
    });

    const carrier: { [key: string]: string } = {};
    propagation.inject(context.active(), carrier);

    const traceContext = new TraceContext({
      traceparent: carrier['traceparent'] || '',
      tracestate: carrier['tracestate'] || '',
    });

    const frameRequest = new WebSocketFrameRequest({
      type: 'frame',
      request: request,
      traceContext: traceContext,
    });

    const transportFormat = streamConfigService.getTransportFormat();
    if (transportFormat === 'binary') {
      this.ws.send(frameRequest.toBinary());
    } else {
      this.ws.send(frameRequest.toJsonString());
    }

    logger.debug('Frame sent via WebSocket with ProcessingConfig', {
      width: image.getWidth(),
      height: image.getHeight(),
      aspectRatio: image.getAspectRatio(),
      filterCount: filters.length,
      filterTypes: filters.map(f => f.getType()).join(','),
      accelerator: accelerator.toString(),
      grayscale: grayscale.toString(),
    });
  }

  sendSingleFrame(
    base64Data: string,
    width: number,
    height: number,
    filters: string[],
    accelerator: string,
    grayscaleType: string,
    blurParams?: Record<string, any>
  ): Promise<WebSocketFrameResponse> {
    return telemetryService.withSpanAsync(
      'WebSocket.sendSingleFrame',
      {
        'image.width': width,
        'image.height': height,
        filters: filters.join(','),
        accelerator: accelerator,
        grayscale_type: grayscaleType,
      },
      async (span) => {
        return new Promise((resolve, reject) => {
          if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket not connected'));
            return;
          }

          span?.addEvent('Preparing WebSocket frame');
          const originalCallback = this.onFrameResultCallback;

          this.onFrameResultCallback = (data: WebSocketFrameResponse) => {
            this.onFrameResultCallback = originalCallback;

            if (data.success && data.response) {
              span?.addEvent('Frame processed successfully');
              span?.setAttribute('result.width', data.response.width);
              span?.setAttribute('result.height', data.response.height);
              resolve(data);
            } else {
              const error = new Error(data.error || 'Unknown error');
              reject(error);
            }
          };

          span?.addEvent('Sending frame via WebSocket');
          this.sendFrame(base64Data, width, height, filters, accelerator, grayscaleType, blurParams);
        });
      }
    );
  }

  onFrameResult(callback: FrameResultCallback): void {
    this.onFrameResultCallback = callback;
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
      this.statsManager.updateWebSocketStatus('disconnected', 'Disconnected');
    }
  }

  sendStartVideo(
    videoId: string,
    filters: string[],
    accelerator: string,
    grayscaleType: string,
    blurParams?: Record<string, any>
  ): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      logger.error('WebSocket not connected');
      return;
    }

    const span = telemetryService.createSpan('sendStartVideo');

    try {
      const filterTypes = filters.map((f) => {
        if (f === 'none') return FilterType.NONE;
        if (f === 'grayscale') return FilterType.GRAYSCALE;
        if (f === 'blur') return FilterType.BLUR;
        return FilterType.NONE;
      });

      const acceleratorType = accelerator === 'gpu' ? AcceleratorType.CUDA : AcceleratorType.CPU;
      const grayscaleTypeEnum =
        grayscaleType === 'bt709' ? GrayscaleType.BT709 : GrayscaleType.BT601;

      let protoBlurParams: GaussianBlurParameters | undefined = undefined;
      if (filters.includes('blur')) {
        if (blurParams) {
          const kernelSize = blurParams.kernel_size !== undefined ? blurParams.kernel_size : 5;
          const sigma = blurParams.sigma !== undefined ? blurParams.sigma : 1.0;
          const separable = blurParams.separable !== undefined ? blurParams.separable : true;
          
          let borderMode = BorderMode.REFLECT;
          if (blurParams.border_mode !== undefined) {
            const borderModeStr = String(blurParams.border_mode).toUpperCase();
            switch (borderModeStr) {
              case 'CLAMP':
                borderMode = BorderMode.CLAMP;
                break;
              case 'REFLECT':
                borderMode = BorderMode.REFLECT;
                break;
              case 'WRAP':
                borderMode = BorderMode.WRAP;
                break;
              default:
                borderMode = BorderMode.REFLECT;
            }
          }

          protoBlurParams = new GaussianBlurParameters({
            kernelSize,
            sigma,
            borderMode,
            separable,
          });
        } else {
          protoBlurParams = new GaussianBlurParameters({
            kernelSize: 5,
            sigma: 1.0,
            borderMode: BorderMode.REFLECT,
            separable: true,
          });
        }
      }

      const startVideoRequest = new StartVideoPlaybackRequest({
        videoId,
        filters: filterTypes,
        accelerator: acceleratorType,
        grayscaleType: grayscaleTypeEnum,
        blurParams: protoBlurParams,
      });

      const frameRequest = new WebSocketFrameRequest({
        type: 'start_video',
        startVideoRequest,
      });

      const transportFormat = streamConfigService.getTransportFormat();
      let messageData: string | Uint8Array;

      if (transportFormat === 'binary') {
        messageData = frameRequest.toBinary();
      } else {
        messageData = frameRequest.toJsonString();
      }

      this.ws.send(messageData);
      logger.debug('Start video message sent', {
        'video.id': videoId,
      });
    } finally {
      if (span) {
        span.end();
      }
    }
  }

  sendStopVideo(videoId: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      logger.error('WebSocket not connected');
      return;
    }

    const span = telemetryService.createSpan('sendStopVideo');

    try {
      const stopVideoRequest = new StopVideoPlaybackRequest({
        sessionId: videoId,
      });

      const frameRequest = new WebSocketFrameRequest({
        type: 'stop_video',
        stopVideoRequest,
      });

      const transportFormat = streamConfigService.getTransportFormat();
      let messageData: string | Uint8Array;

      if (transportFormat === 'binary') {
        messageData = frameRequest.toBinary();
      } else {
        messageData = frameRequest.toJsonString();
      }

      this.ws.send(messageData);
      logger.debug('Stop video message sent', {
        'video.id': videoId,
      });
    } finally {
      if (span) {
        span.end();
      }
    }
  }
}
