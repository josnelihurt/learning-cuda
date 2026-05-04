import type { ProcessImageResponse } from '@/gen/image_processor_service_pb';
import type { ImageData, FilterData, GrayscaleAlgorithm } from '@/domain/value-objects';
import type { AcceleratorType } from '@/gen/common_pb';

export interface IFrameTransportService {
  connect(): void;
  disconnect(): void;
  sendFrame(imageData: string, width: number, height: number, filters: FilterData[], accelerator: AcceleratorType): void;
  sendFrameWithImageData(image: ImageData, filters: FilterData[], accelerator: AcceleratorType): void;
  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: AcceleratorType): void;
  sendFrameWithProcessingConfig(image: ImageData, filters: FilterData[], accelerator: AcceleratorType, grayscale: GrayscaleAlgorithm): void;
  sendSingleFrame(imageData: string, width: number, height: number, filters: FilterData[], accelerator: AcceleratorType): Promise<ProcessImageResponse>;
  onFrameResult(callback: (data: ProcessImageResponse) => void): void;
  isConnected(): boolean;
  getConnectionStatus(): { state: 'connected' | 'disconnected' | 'connecting' | 'error'; lastRequest: string | null; lastRequestTime: Date | null };
}
