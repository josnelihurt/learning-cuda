import type { ProcessImageResponse } from '@/gen/image_processor_service_pb';
import type { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '@/domain/value-objects';

export interface IFrameTransportService {
  connect(): void;
  disconnect(): void;
  sendFrame(imageData: string, width: number, height: number, filters: FilterData[], accelerator: string): void;
  sendFrameWithImageData(image: ImageData, filters: FilterData[], accelerator: string): void;
  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: string): void;
  sendFrameWithProcessingConfig(image: ImageData, filters: FilterData[], accelerator: AcceleratorConfig, grayscale: GrayscaleAlgorithm): void;
  sendSingleFrame(imageData: string, width: number, height: number, filters: FilterData[], accelerator: string): Promise<ProcessImageResponse>;
  sendStartVideo(videoId: string, filters: FilterData[], accelerator: string): void;
  sendStopVideo(videoId?: string): void;
  onFrameResult(callback: (data: ProcessImageResponse) => void): void;
  isConnected(): boolean;
  getConnectionStatus(): { state: 'connected' | 'disconnected' | 'connecting' | 'error'; lastRequest: string | null; lastRequestTime: Date | null };
}
