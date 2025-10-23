import type { WebSocketFrameResponse } from '../../gen/image_processor_service_pb';
import type { ImageData, FilterData } from '../value-objects';

export interface IWebSocketService {
  connect(): void;
  disconnect(): void;
  sendFrame(imageData: string, width: number, height: number, filters: string[], accelerator: string, grayscaleType: string): void;
  sendFrameWithImageData(image: ImageData, filters: string[], accelerator: string, grayscaleType: string): void;
  sendFrameWithValueObjects(image: ImageData, filters: FilterData[], accelerator: string, grayscaleType: string): void;
  sendSingleFrame(imageData: string, width: number, height: number, filters: string[], accelerator: string, grayscaleType: string): Promise<WebSocketFrameResponse>;
  sendStartVideo(videoId: string, filters: string[], accelerator: string, grayscaleType: string): void;
  sendStopVideo(videoId?: string): void;
  onFrameResult(callback: (data: WebSocketFrameResponse) => void): void;
  isConnected(): boolean;
}
