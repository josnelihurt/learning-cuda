import { createPromiseClient, PromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { FileService } from '../gen/file_service_connect';
import { StaticVideo } from '../gen/common_pb';
import { telemetryService } from './telemetry-service';
import { logger } from './otel-logger';
import type { IVideoService } from '../domain/interfaces/IVideoService';

const tracingInterceptor: Interceptor = (next) => async (req) => {
  const headers = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(headers)) {
    req.header.set(key, value);
  }
  return await next(req);
};

class VideoService implements IVideoService {
  private fileClient: PromiseClient<typeof FileService>;
  private videos: StaticVideo[] = [];
  private initPromise: Promise<void> | null = null;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      interceptors: [tracingInterceptor],
    });

    this.fileClient = createPromiseClient(FileService, transport);
  }

  async initialize(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = telemetryService.withSpanAsync(
      'VideoService.initialize',
      {
        'http.method': 'POST',
        'rpc.service': 'FileService',
        'rpc.method': 'listAvailableVideos',
      },
      async (span) => {
        try {
          span?.addEvent('Fetching available videos');
          const response = await this.fileClient.listAvailableVideos({});

          this.videos = response.videos || [];

          span?.setAttribute('videos.count', this.videos.length);
          span?.setAttribute('videos.loaded', true);
          span?.addEvent('Videos loaded successfully');

          logger.debug('Available videos loaded', {
            'videos.count': this.videos.length,
          });
        } catch (error) {
          span?.addEvent('Failed to load videos');
          span?.setAttribute('error', true);

          logger.error('Failed to load videos', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          this.videos = [];
        }
      }
    );

    return this.initPromise;
  }

  getVideos(): StaticVideo[] {
    return this.videos;
  }

  getDefaultVideo(): StaticVideo | undefined {
    return this.videos.find((vid) => vid.isDefault);
  }

  getById(id: string): StaticVideo | undefined {
    return this.videos.find((vid) => vid.id === id);
  }

  isInitialized(): boolean {
    return this.videos.length > 0;
  }

  async listAvailableVideos(): Promise<StaticVideo[]> {
    return telemetryService.withSpanAsync(
      'VideoService.listAvailableVideos',
      {
        'http.method': 'POST',
        'rpc.service': 'FileService',
        'rpc.method': 'listAvailableVideos',
      },
      async (span) => {
        try {
          span?.addEvent('Fetching available videos');
          const response = await this.fileClient.listAvailableVideos({});

          const videos = response.videos || [];

          span?.setAttribute('videos.count', videos.length);
          span?.addEvent('Videos loaded successfully');

          logger.debug('Available videos', {
            'videos.count': videos.length,
          });
          return videos;
        } catch (error) {
          span?.addEvent('Failed to load videos');
          span?.setAttribute('error', true);

          logger.error('Failed to load videos', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          return [];
        }
      }
    );
  }

  async uploadVideo(file: File): Promise<StaticVideo | null> {
    return telemetryService.withSpanAsync(
      'VideoService.uploadVideo',
      {
        'http.method': 'POST',
        'rpc.service': 'FileService',
        'rpc.method': 'uploadVideo',
        'file.name': file.name,
        'file.size': file.size,
      },
      async (span) => {
        try {
          span?.addEvent('Reading video file');
          const fileData = await file.arrayBuffer();

          span?.addEvent('Uploading video');
          const response = await this.fileClient.uploadVideo({
            fileData: new Uint8Array(fileData),
            filename: file.name,
          });

          if (response.video) {
            span?.setAttribute('video.id', response.video.id);
            span?.setAttribute('upload.success', true);
            span?.addEvent('Video uploaded successfully');

            this.videos.push(response.video);

            return response.video;
          }

          return null;
        } catch (error) {
          span?.addEvent('Video upload failed');
          span?.setAttribute('error', true);

          logger.error('Failed to upload video', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          throw error;
        }
      }
    );
  }
}

export const videoService = new VideoService();
