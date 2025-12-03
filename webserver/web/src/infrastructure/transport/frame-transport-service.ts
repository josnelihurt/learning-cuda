import type { IFrameTransportService } from '../../domain/interfaces/IFrameTransportService';
import type { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '../../domain/value-objects';
import { WebSocketFrameResponse } from '../../gen/image_processor_service_pb';
import { featureFlagsService } from '../external/feature-flags-service';
import { logger } from '../observability/otel-logger';

export class FrameTransportService implements IFrameTransportService {
  private currentTransport: IFrameTransportService | null = null;
  private transportSelectionPromise: Promise<IFrameTransportService> | null = null;

  constructor(
    private wsTransport: IFrameTransportService,
    private grpcTransport: IFrameTransportService,
    private webrtcTransport: IFrameTransportService
  ) {}

  private async selectTransport(): Promise<IFrameTransportService> {
    if (this.transportSelectionPromise) {
      return this.transportSelectionPromise;
    }

    this.transportSelectionPromise = (async () => {
      try {
        const useGRPC = await featureFlagsService.isFeatureEnabled('processor_use_grpc_backend');
        const selected = useGRPC ? this.grpcTransport : this.wsTransport;
        this.currentTransport = selected;
        logger.debug('Transport selected', {
          transport: useGRPC ? 'gRPC' : 'WebSocket',
        });
        return selected;
      } catch (error) {
        logger.error('Failed to check feature flag, defaulting to WebSocket', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
        this.currentTransport = this.wsTransport;
        return this.wsTransport;
      } finally {
        this.transportSelectionPromise = null;
      }
    })();

    return this.transportSelectionPromise;
  }

  private getCurrentTransport(): IFrameTransportService {
    if (this.currentTransport) {
      return this.currentTransport;
    }
    return this.wsTransport;
  }

  connect(): void {
    const transport = this.getCurrentTransport();
    this.selectTransport().then((selectedTransport) => {
      if (selectedTransport !== transport) {
        transport.disconnect();
      }
      selectedTransport.connect();
    }).catch((error) => {
      logger.error('Failed to connect transport', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      transport.connect();
    });
  }

  disconnect(): void {
    const transport = this.getCurrentTransport();
    transport.disconnect();
  }

  sendFrame(
    base64Data: string,
    width: number,
    height: number,
    filters: FilterData[],
    accelerator: string
  ): void {
    const transport = this.getCurrentTransport();
    transport.sendFrame(base64Data, width, height, filters, accelerator);
  }

  sendFrameWithImageData(image: ImageData, filters: FilterData[], accelerator: string): void {
    const transport = this.getCurrentTransport();
    transport.sendFrameWithImageData(image, filters, accelerator);
  }

  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: string): void {
    const transport = this.getCurrentTransport();
    transport.sendFrameWithValueObjects(image, filters, accelerator);
  }

  sendFrameWithProcessingConfig(
    image: ImageData,
    filters: FilterData[],
    accelerator: AcceleratorConfig,
    grayscale: GrayscaleAlgorithm
  ): void {
    const transport = this.getCurrentTransport();
    transport.sendFrameWithProcessingConfig(image, filters, accelerator, grayscale);
  }

  async sendSingleFrame(
    base64Data: string,
    width: number,
    height: number,
    filters: FilterData[],
    accelerator: string
  ): Promise<WebSocketFrameResponse> {
    const transport = await this.selectTransport();
    return transport.sendSingleFrame(base64Data, width, height, filters, accelerator);
  }

  sendStartVideo(videoId: string, filters: FilterData[], accelerator: string): void {
    const transport = this.getCurrentTransport();
    transport.sendStartVideo(videoId, filters, accelerator);
  }

  sendStopVideo(videoId?: string): void {
    const transport = this.getCurrentTransport();
    transport.sendStopVideo(videoId);
  }

  onFrameResult(callback: (data: WebSocketFrameResponse) => void): void {
    this.wsTransport.onFrameResult(callback);
    this.grpcTransport.onFrameResult(callback);
  }

  isConnected(): boolean {
    const transport = this.getCurrentTransport();
    return transport.isConnected();
  }

  getConnectionStatus(): { state: 'connected' | 'disconnected' | 'connecting' | 'error'; lastRequest: string | null; lastRequestTime: Date | null } {
    const transport = this.getCurrentTransport();
    return transport.getConnectionStatus();
  }
}

