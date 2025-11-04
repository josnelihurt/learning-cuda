import type { WebSocketFrameResponse } from '../../gen/image_processor_service_pb';
import type { ImageData, FilterData, AcceleratorConfig, GrayscaleAlgorithm } from '../value-objects';

export interface IWebSocketService {
  connect(): void;
  disconnect(): void;
  sendFrame(imageData: string, width: number, height: number, filters: string[], accelerator: string, grayscaleType: string, blurParams?: Record<string, any>): void;
  sendFrameWithImageData(image: ImageData, filters: string[], accelerator: string, grayscaleType: string, blurParams?: Record<string, any>): void;
  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: string, grayscaleType: string): void;
  sendFrameWithProcessingConfig(image: ImageData, filters: FilterData[], accelerator: AcceleratorConfig, grayscale: GrayscaleAlgorithm): void;
  sendSingleFrame(imageData: string, width: number, height: number, filters: string[], accelerator: string, grayscaleType: string, blurParams?: Record<string, any>): Promise<WebSocketFrameResponse>;
  sendStartVideo(videoId: string, filters: string[], accelerator: string, grayscaleType: string, blurParams?: Record<string, any>): void;
  sendStopVideo(videoId?: string): void;
  onFrameResult(callback: (data: WebSocketFrameResponse) => void): void;
  isConnected(): boolean;
}
