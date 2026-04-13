import type { IFrameTransportService } from '../domain/interfaces/IFrameTransportService';
import type { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '../domain/value-objects';
import { WebSocketFrameResponse } from '../gen/image_processor_service_pb';
import type { StatsPanel } from '../components/app/stats-panel';
import type { CameraPreview } from '../components/video/camera-preview';
import type { ToastContainer } from '../components/app/toast-container';

export class WebRTCFrameTransportService implements IFrameTransportService {
  constructor(
    private statsManager: StatsPanel,
    private cameraManager: CameraPreview,
    private toastManager: ToastContainer
  ) {}

  connect(): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  disconnect(): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  sendFrame(
    base64Data: string,
    width: number,
    height: number,
    filters: FilterData[],
    accelerator: string
  ): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  sendFrameWithImageData(image: ImageData, filters: FilterData[], accelerator: string): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: string): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  sendFrameWithProcessingConfig(
    image: ImageData,
    filters: FilterData[],
    accelerator: AcceleratorConfig,
    grayscale: GrayscaleAlgorithm
  ): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  async sendSingleFrame(
    base64Data: string,
    width: number,
    height: number,
    filters: FilterData[],
    accelerator: string
  ): Promise<WebSocketFrameResponse> {
    throw new Error('WebRTC frame transport not implemented');
  }

  sendStartVideo(videoId: string, filters: FilterData[], accelerator: string): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  sendStopVideo(videoId?: string): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  onFrameResult(callback: (data: WebSocketFrameResponse) => void): void {
    throw new Error('WebRTC frame transport not implemented');
  }

  isConnected(): boolean {
    return false;
  }

  getConnectionStatus(): { state: 'connected' | 'disconnected' | 'connecting' | 'error'; lastRequest: string | null; lastRequestTime: Date | null } {
    return {
      state: 'disconnected',
      lastRequest: null,
      lastRequestTime: null,
    };
  }
}

