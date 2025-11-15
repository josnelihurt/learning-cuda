import type { WebSocketFrameResponse } from '../../gen/image_processor_service_pb';
import type { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '../value-objects';

export interface IWebSocketService {
  connect(): void;
  disconnect(): void;
  sendFrame(imageData: string, width: number, height: number, filters: FilterData[], accelerator: string): void;
  sendFrameWithImageData(image: ImageData, filters: FilterData[], accelerator: string): void;
  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: string): void;
  sendFrameWithProcessingConfig(image: ImageData, filters: FilterData[], accelerator: AcceleratorConfig, grayscale: GrayscaleAlgorithm): void;
  sendSingleFrame(imageData: string, width: number, height: number, filters: FilterData[], accelerator: string): Promise<WebSocketFrameResponse>;
  sendStartVideo(videoId: string, filters: FilterData[], accelerator: string): void;
  sendStopVideo(videoId?: string): void;
  onFrameResult(callback: (data: WebSocketFrameResponse) => void): void;
  isConnected(): boolean;
}
