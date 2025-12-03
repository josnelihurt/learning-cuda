import { createPromiseClient, PromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ImageProcessorService } from '../../gen/image_processor_service_connect';
import { ProcessImageRequest, ProcessImageResponse } from '../../gen/image_processor_service_pb';
import { WebSocketFrameResponse } from '../../gen/image_processor_service_pb';
import type { IFrameTransportService } from '../../domain/interfaces/IFrameTransportService';
import { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '../../domain/value-objects';
import type { StatsPanel } from '../../components/app/stats-panel';
import type { CameraPreview } from '../../components/video/camera-preview';
import type { ToastContainer } from '../../components/app/toast-container';
import { telemetryService } from '../observability/telemetry-service';
import { logger } from '../observability/otel-logger';
import { context, propagation } from '@opentelemetry/api';
import { TraceContext, AcceleratorType, GaussianBlurParameters, BorderMode } from '../../gen/common_pb';
import { GenericFilterSelection, GenericFilterParameterSelection } from '../../gen/image_processor_service_pb';
import { StartVideoPlaybackRequest, StopVideoPlaybackRequest } from '../../gen/image_processor_service_pb';

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

function extractBlurParams(filters: FilterData[]): GaussianBlurParameters | undefined {
  const blurFilter = filters.find(f => f.isBlur());
  if (!blurFilter) {
    return undefined;
  }

  const params = blurFilter.getParameters();
  
  let kernelSize = 5;
  if (params.kernel_size !== undefined) {
    const kernelSizeValue = typeof params.kernel_size === 'string' 
      ? parseInt(params.kernel_size, 10) 
      : params.kernel_size;
    kernelSize = !isNaN(kernelSizeValue) && kernelSizeValue > 0 ? kernelSizeValue : 5;
    if (kernelSize % 2 === 0) {
      kernelSize += 1;
    }
  }
  
  let sigma = 1.0;
  if (params.sigma !== undefined) {
    const sigmaValue = typeof params.sigma === 'string' 
      ? parseFloat(params.sigma) 
      : params.sigma;
    sigma = !isNaN(sigmaValue) && sigmaValue >= 0 ? sigmaValue : 1.0;
  }
  
  let separable = true;
  if (params.separable !== undefined) {
    if (typeof params.separable === 'string') {
      separable = params.separable === 'true' || params.separable === '1';
    } else {
      separable = Boolean(params.separable);
    }
  }
  
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

type FrameResultCallback = (data: WebSocketFrameResponse) => void;

export class GRPCFrameTransportService implements IFrameTransportService {
  private client: PromiseClient<typeof ImageProcessorService>;
  private stream: any | null = null;
  private onFrameResultCallback: FrameResultCallback | null = null;
  private connectionState: 'connected' | 'disconnected' | 'connecting' | 'error' = 'disconnected';
  private lastRequest: string | null = null;
  private lastRequestTime: Date | null = null;
  private reconnectTimeout = 3000;
  private reconnectTimer: number | null = null;

  constructor(
    private statsManager: StatsPanel,
    private cameraManager: CameraPreview,
    private toastManager: ToastContainer
  ) {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
    });
    this.client = createPromiseClient(ImageProcessorService, transport);
  }

  connect(): void {
    if (this.connectionState === 'connected' || this.connectionState === 'connecting') {
      return;
    }

    this.connectionState = 'connecting';
    this.statsManager.updateWebSocketStatus('connecting', 'Connecting...');

    telemetryService.withSpanAsync(
      'GRPCFrameTransport.connect',
      {},
      async (span) => {
        try {
          span?.addEvent('Creating bidirectional stream');
          this.stream = (this.client as any).streamProcessVideo();

          span?.addEvent('Setting up response handler');
          this.stream.responses.onMessage((response: ProcessImageResponse) => {
            this.handleResponse(response);
          });

          this.stream.responses.onError((error: Error) => {
            logger.error('gRPC stream error', {
              'error.message': error.message,
            });
            this.connectionState = 'error';
            this.statsManager.updateWebSocketStatus('disconnected', 'Stream error');
            this.toastManager.warning('gRPC Error', 'Connection error, attempting to reconnect...');
            this.scheduleReconnect();
          });

          this.stream.responses.onClose(() => {
            logger.info('gRPC stream closed - Reconnecting...');
            this.connectionState = 'disconnected';
            this.statsManager.updateWebSocketStatus('connecting', 'Reconnecting...');
            this.scheduleReconnect();
          });

          this.connectionState = 'connected';
          this.statsManager.updateWebSocketStatus('connected', 'Connected');
          logger.info('gRPC stream connected');
          span?.addEvent('Stream connected successfully');
        } catch (error) {
          span?.setAttribute('error', true);
          span?.setAttribute('error.message', error instanceof Error ? error.message : String(error));
          logger.error('Failed to connect gRPC stream', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          this.connectionState = 'error';
          this.statsManager.updateWebSocketStatus('disconnected', 'Connection failed');
          this.scheduleReconnect();
        }
      }
    );
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    this.reconnectTimer = window.setTimeout(() => {
      this.connect();
    }, this.reconnectTimeout);
  }

  private handleResponse(response: ProcessImageResponse): void {
    const receiveTime = performance.now();
    this.cameraManager.setProcessing(false);

    try {
      const wsResponse = new WebSocketFrameResponse({
        type: 'frame_result',
        success: response.code === 0,
        error: response.code !== 0 ? response.message : undefined,
        response: response,
      });

      if (response.code === 0) {
        const sendTime = this.cameraManager.getLastFrameTime();
        const processingTime = receiveTime - sendTime;
        this.statsManager.updateProcessingStats(processingTime);

        if (this.onFrameResultCallback) {
          this.onFrameResultCallback(wsResponse);
        }
      } else {
        logger.error('Frame processing error', {
          'error.message': response.message || 'Unknown error',
        });
        this.toastManager.error('Processing Error', response.message || 'Unknown error');
        this.statsManager.updateCameraStatus('Processing failed', 'error');
      }
    } catch (error) {
      logger.error('Error handling gRPC response', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
    }
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.stream) {
      this.stream.requests.close().catch((error: Error) => {
        logger.error('Error closing gRPC stream', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      });
      this.stream = null;
    }

    this.connectionState = 'disconnected';
    this.statsManager.updateWebSocketStatus('disconnected', 'Disconnected');
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
    if (!this.stream || this.connectionState !== 'connected') {
      logger.warn('gRPC stream not connected, cannot send frame');
      return;
    }

    this.cameraManager.setProcessing(true);
    this.lastRequest = 'SendFrame';
    this.lastRequestTime = new Date();

    const protoAccelerator = accelerator === 'cpu' ? AcceleratorType.CPU : AcceleratorType.CUDA;
    const genericFilters = toGenericFilterSelections(filters);

    const imageDataB64 = image.getBase64().replace(/^data:image\/(png|jpeg);base64,/, '');
    const imageBytes = Uint8Array.from(atob(imageDataB64), (c) => c.charCodeAt(0));

    const carrier: { [key: string]: string } = {};
    propagation.inject(context.active(), carrier);

    const traceContext = new TraceContext({
      traceparent: carrier['traceparent'] || '',
      tracestate: carrier['tracestate'] || '',
    });

    const request = new ProcessImageRequest({
      imageData: imageBytes,
      width: image.getWidth(),
      height: image.getHeight(),
      channels: 3,
      accelerator: protoAccelerator,
      genericFilters,
      traceContext,
    });

    this.stream.requests.send(request).catch((error: Error) => {
      logger.error('Failed to send frame via gRPC', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      this.cameraManager.setProcessing(false);
    });

    logger.debug('Frame sent via gRPC', {
      width: image.getWidth(),
      height: image.getHeight(),
      filterCount: filters.length,
    });
  }

  sendFrameWithProcessingConfig(
    image: ImageData,
    filters: FilterData[],
    accelerator: AcceleratorConfig,
    grayscale: GrayscaleAlgorithm
  ): void {
    if (!this.stream || this.connectionState !== 'connected') {
      logger.warn('gRPC stream not connected, cannot send frame');
      return;
    }

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

    const carrier: { [key: string]: string } = {};
    propagation.inject(context.active(), carrier);

    const traceContext = new TraceContext({
      traceparent: carrier['traceparent'] || '',
      tracestate: carrier['tracestate'] || '',
    });

    const request = new ProcessImageRequest({
      imageData: imageBytes,
      width: image.getWidth(),
      height: image.getHeight(),
      channels: 3,
      accelerator: protoAccelerator,
      genericFilters,
      traceContext,
    });

    this.stream.requests.send(request).catch((error: Error) => {
      logger.error('Failed to send frame via gRPC', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      this.cameraManager.setProcessing(false);
    });
  }

  async sendSingleFrame(
    base64Data: string,
    width: number,
    height: number,
    filters: FilterData[],
    accelerator: string
  ): Promise<WebSocketFrameResponse> {
    return telemetryService.withSpanAsync(
      'GRPCFrameTransport.sendSingleFrame',
      {
        'image.width': width,
        'image.height': height,
        filters: filters.map((f) => f.getId()).join(','),
        accelerator: accelerator,
      },
      async (span) => {
        return new Promise((resolve, reject) => {
          if (!this.stream || this.connectionState !== 'connected') {
            reject(new Error('gRPC stream not connected'));
            return;
          }

          span?.addEvent('Preparing gRPC frame');
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

          span?.addEvent('Sending frame via gRPC');
          this.sendFrame(base64Data, width, height, filters, accelerator);
        });
      }
    );
  }

  onFrameResult(callback: FrameResultCallback): void {
    this.onFrameResultCallback = callback;
  }

  isConnected(): boolean {
    return this.stream !== null && this.connectionState === 'connected';
  }

  getConnectionStatus(): { state: 'connected' | 'disconnected' | 'connecting' | 'error'; lastRequest: string | null; lastRequestTime: Date | null } {
    return {
      state: this.connectionState,
      lastRequest: this.lastRequest,
      lastRequestTime: this.lastRequestTime,
    };
  }

  sendStartVideo(videoId: string, filters: FilterData[], accelerator: string): void {
    logger.warn('sendStartVideo not implemented for gRPC transport');
  }

  sendStopVideo(videoId?: string): void {
    logger.warn('sendStopVideo not implemented for gRPC transport');
  }
}

