import { createPromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { FileService } from '../../gen/file_service_connect';
import type { StaticImage } from '../../gen/config_service_pb';
import { telemetryService } from '../observability/telemetry-service';
import { logger } from '../observability/otel-logger';
import type { IFileService } from '../../domain/interfaces/IFileService';

class FileServiceClient implements IFileService {
  private client;
  private isInit = false;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      useHttpGet: true,
    });
    this.client = createPromiseClient(FileService, transport);
  }

  async initialize(): Promise<void> {
    if (this.isInit) {
      return;
    }

    const span = telemetryService.createSpan('FileService.initialize');
    try {
      logger.debug('File service initialized');
      this.isInit = true;
      span?.end();
    } catch (error) {
      span?.recordException(error as Error);
      span?.end();
      throw error;
    }
  }

  isInitialized(): boolean {
    return this.isInit;
  }

  async listAvailableImages(): Promise<StaticImage[]> {
    const span = telemetryService.createSpan('FileService.listAvailableImages');
    try {
      const response = await this.client.listAvailableImages({});
      span?.setAttribute('images.count', response.images.length);
      span?.end();
      return response.images;
    } catch (error) {
      span?.recordException(error as Error);
      span?.end();
      throw error;
    }
  }

  async uploadImage(file: File): Promise<StaticImage> {
    const span = telemetryService.createSpan('FileService.uploadImage');
    span?.setAttribute('filename', file.name);
    span?.setAttribute('file_size', file.size);

    try {
      const fileData = await this.readFileAsBytes(file);

      const response = await this.client.uploadImage({
        filename: file.name,
        fileData: fileData,
      });

      if (!response.image) {
        throw new Error('Upload failed: No image returned');
      }

      span?.setAttribute('image.id', response.image.id);
      span?.setAttribute('upload.message', response.message);
      span?.end();

      logger.info('Image uploaded successfully', {
        'image.id': response.image.id,
      });
      return response.image;
    } catch (error) {
      span?.recordException(error as Error);
      span?.end();
      throw error;
    }
  }

  private async readFileAsBytes(file: File): Promise<Uint8Array> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const arrayBuffer = reader.result as ArrayBuffer;
        resolve(new Uint8Array(arrayBuffer));
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsArrayBuffer(file);
    });
  }
}

export const fileService = new FileServiceClient();
