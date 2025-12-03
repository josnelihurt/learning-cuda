import type { StatsPanel } from '../../components/app/stats-panel';
import type { CameraPreview } from '../../components/video/camera-preview';
import type { ToastContainer } from '../../components/app/toast-container';
import {
  WebSocketFrameRequest,
  WebSocketFrameResponse,
  ProcessImageRequest,
  StartVideoPlaybackRequest,
  StopVideoPlaybackRequest,
  GenericFilterSelection,
  GenericFilterParameterSelection,
} from '../../gen/image_processor_service_pb';
import { FilterType, AcceleratorType, GrayscaleType, TraceContext, GaussianBlurParameters, BorderMode } from '../../gen/common_pb';
import { streamConfigService } from '../../application/services/config-service';
import { telemetryService } from '../observability/telemetry-service';
import { logger } from '../observability/otel-logger';
import { context, propagation } from '@opentelemetry/api';
import type { IWebSocketService } from '../../domain/interfaces/IWebSocketService';
import type { IFrameTransportService } from '../../domain/interfaces/IFrameTransportService';
import { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '../../domain/value-objects';

function toGenericFilterSelections(filters: FilterData[]): GenericFilterSelection[] {
  return filters.map((filter) => {
    const params = filter.getParameters();
    const parameterSelections = Object.entries(params)
      .map(([parameterId, value]) => {
        const values = serializeParameterValues(value);
        if (values.length === 0) {
          return null;
        }
        return new GenericFilterParameterSelection({
          parameterId,
          values,
        });
      })
      .filter((selection): selection is GenericFilterParameterSelection => selection !== null);

    return new GenericFilterSelection({
      filterId: filter.getId(),
      parameters: parameterSelections,
    });
  });
}

function serializeParameterValues(value: unknown): string[] {
  if (value === undefined || value === null) {
    return [];
  }

  if (Array.isArray(value)) {
    return value
      .map((entry) => serializeSingleValue(entry))
      .filter((entry) => entry !== '');
  }

  const serialized = serializeSingleValue(value);
  return serialized ? [serialized] : [];
}

function serializeSingleValue(value: unknown): string {
  if (value === undefined || value === null) {
    return '';
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  return String(value);
}

type FrameResultCallback = (data: WebSocketFrameResponse) => void;

function extractBlurParams(filters: FilterData[]): GaussianBlurParameters | undefined {
  const blurFilter = filters.find(f => f.isBlur());
  if (!blurFilter) {
    return undefined;
  }

  const params = blurFilter.getParameters();
  
  // Convert kernel_size to number, handling both string and number inputs
  let kernelSize = 5;
  if (params.kernel_size !== undefined) {
    const kernelSizeValue = typeof params.kernel_size === 'string' 
      ? parseInt(params.kernel_size, 10) 
      : params.kernel_size;
    kernelSize = !isNaN(kernelSizeValue) && kernelSizeValue > 0 ? kernelSizeValue : 5;
    // Ensure it's odd
    if (kernelSize % 2 === 0) {
      kernelSize += 1;
    }
  }
  
  // Convert sigma to number, handling both string and number inputs
  let sigma = 1.0;
  if (params.sigma !== undefined) {
    const sigmaValue = typeof params.sigma === 'string' 
      ? parseFloat(params.sigma) 
      : params.sigma;
    sigma = !isNaN(sigmaValue) && sigmaValue >= 0 ? sigmaValue : 1.0;
  }
  
  // Convert separable to boolean, handling both string and boolean inputs
  let separable = true;
  if (params.separable !== undefined) {
    if (typeof params.separable === 'string') {
      separable = params.separable === 'true' || params.separable === '1';
    } else {
      separable = Boolean(params.separable);
    }
  }
  
  // Convert border_mode to enum
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
export class WebSocketService implements IWebSocketService, IFrameTransportService {
  private ws: WebSocket | null = null;
  private reconnectTimeout = 3000;
  private onFrameResultCallback: FrameResultCallback | null = null;
  private lastRequest: string | null = null;
  private lastRequestTime: Date | null = null;
  private connectionState: 'connected' | 'disconnected' | 'connecting' | 'error' = 'disconnected';

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
      this.connectionState = 'connected';
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
      this.connectionState = 'error';
      logger.error('WebSocket error', {
        'error.type': error.type,
      });
      this.toastManager.warning('WebSocket Error', 'Connection error, attempting to reconnect...');
      this.statsManager.updateWebSocketStatus('disconnected', 'Connection error');
      this.cameraManager.setProcessing(false);
    };

    this.ws.onclose = () => {
      this.connectionState = 'connecting';
      logger.info('WebSocket closed - Reconnecting...');
      this.statsManager.updateWebSocketStatus('connecting', 'Reconnecting...');
      setTimeout(() => this.connect(), this.reconnectTimeout);
    };
  }

  sendFrame(
    base64Data: string,
    width: number,
    height: number,
    filters: FilterData[],
    accelerator: string
  ): void {
    const image = new ImageData(base64Data, width, height);
    this.sendFrameWithImageData(image, filters, accelerator);
  }

  sendFrameWithImageData(image: ImageData, filters: FilterData[], accelerator: string): void {
    this.sendFrameWithValueObjects(image, filters, accelerator);
  }

  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: string): void {
    if (!this.ws || this.ws.readyState !== 1) return; // WebSocket.OPEN

    this.cameraManager.setProcessing(true);
    this.lastRequest = 'SendFrame';
    this.lastRequestTime = new Date();

    const protoAccelerator = accelerator === 'cpu' ? AcceleratorType.CPU : AcceleratorType.CUDA;
    const genericFilters = toGenericFilterSelections(filters);

    const imageDataB64 = image.getBase64().replace(/^data:image\/(png|jpeg);base64,/, '');
    const imageBytes = Uint8Array.from(atob(imageDataB64), (c) => c.charCodeAt(0));

    const request = new ProcessImageRequest({
      imageData: imageBytes,
      width: image.getWidth(),
      height: image.getHeight(),
      channels: 3,
      accelerator: protoAccelerator,
      genericFilters,
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

    const protoAccelerator = accelerator.toProtocol();
    const genericFilters = toGenericFilterSelections(filters);
    if (!genericFilters.some((selection) => selection.filterId === 'grayscale')) {
      genericFilters.push(
        new GenericFilterSelection({
          filterId: 'grayscale',
          parameters: [
            new GenericFilterParameterSelection({
              parameterId: 'algorithm',
              values: [grayscale.toString()],
            }),
          ],
        })
      );
    }

    const imageDataB64 = image.getBase64().replace(/^data:image\/(png|jpeg);base64,/, '');
    const imageBytes = Uint8Array.from(atob(imageDataB64), (c) => c.charCodeAt(0));

    const request = new ProcessImageRequest({
      imageData: imageBytes,
      width: image.getWidth(),
      height: image.getHeight(),
      channels: 3,
      accelerator: protoAccelerator,
      genericFilters,
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
    filters: FilterData[],
    accelerator: string
  ): Promise<WebSocketFrameResponse> {
    return telemetryService.withSpanAsync(
      'WebSocket.sendSingleFrame',
      {
        'image.width': width,
        'image.height': height,
        filters: filters.map((f) => f.getId()).join(','),
        accelerator: accelerator,
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
          this.sendFrame(base64Data, width, height, filters, accelerator);
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

  getConnectionStatus(): { state: 'connected' | 'disconnected' | 'connecting' | 'error'; lastRequest: string | null; lastRequestTime: Date | null } {
    return {
      state: this.connectionState,
      lastRequest: this.lastRequest,
      lastRequestTime: this.lastRequestTime,
    };
  }

  disconnect(): void {
    if (this.ws) {
      this.connectionState = 'disconnected';
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
      this.statsManager.updateWebSocketStatus('disconnected', 'Disconnected');
    }
  }

  sendStartVideo(
    videoId: string,
    filters: FilterData[],
    accelerator: string
  ): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      logger.error('WebSocket not connected');
      return;
    }

    const span = telemetryService.createSpan('sendStartVideo');

    try {
      const filterTypes = filters.map((f) => f.toProtocol());
      const acceleratorType = accelerator === 'gpu' ? AcceleratorType.CUDA : AcceleratorType.CPU;
      const grayscaleTypeEnum = this.mapGrayscaleToProtocol(this.getGrayscaleAlgorithm(filters));

      let protoBlurParams = extractBlurParams(filters);
      if (!protoBlurParams && filters.some((f) => f.isBlur())) {
        protoBlurParams = new GaussianBlurParameters({
          kernelSize: 5,
          sigma: 1.0,
          borderMode: BorderMode.REFLECT,
          separable: true,
        });
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

  private getGrayscaleAlgorithm(filters: FilterData[]): string {
    const grayscaleFilter = filters.find((filter) => filter.isGrayscale());
    if (!grayscaleFilter) {
      return 'bt601';
    }
    const params = grayscaleFilter.getParameters();
    return (params.algorithm as string) || 'bt601';
  }

  private mapGrayscaleToProtocol(algorithm: string): GrayscaleType {
    switch ((algorithm || '').toLowerCase()) {
      case 'bt709':
        return GrayscaleType.BT709;
      case 'average':
        return GrayscaleType.AVERAGE;
      case 'lightness':
        return GrayscaleType.LIGHTNESS;
      case 'luminosity':
        return GrayscaleType.LUMINOSITY;
      case 'bt601':
      default:
        return GrayscaleType.BT601;
    }
  }
}
