import { createPromiseClient, PromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService } from '../gen/config_service_connect';
import { FileService } from '../gen/file_service_connect';
import { InputSource } from '../gen/config_service_pb';
import { StaticImage, StaticVideo } from '../gen/common_pb';
import { telemetryService } from './telemetry-service';
import { logger } from './otel-logger';

const tracingInterceptor: Interceptor = (next) => async (req) => {
    const headers = telemetryService.getTraceHeaders();
    for (const [key, value] of Object.entries(headers)) {
        req.header.set(key, value);
    }
    return await next(req);
};

class InputSourceService {
    private configClient: PromiseClient<typeof ConfigService>;
    private fileClient: PromiseClient<typeof FileService>;
    private sources: InputSource[] = [];
    private initPromise: Promise<void> | null = null;

    constructor() {
        const transport = createConnectTransport({
            baseUrl: window.location.origin,
            interceptors: [tracingInterceptor],
        });

        this.configClient = createPromiseClient(ConfigService, transport);
        this.fileClient = createPromiseClient(FileService, transport);
    }

    async initialize(): Promise<void> {
        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = telemetryService.withSpanAsync(
            'InputSourceService.initialize',
            {
                'http.method': 'POST',
                'rpc.service': 'ConfigService',
                'rpc.method': 'listInputs',
            },
            async (span) => {
                try {
                    span?.addEvent('Fetching input sources');
                    const response = await this.configClient.listInputs({});
                    
                    this.sources = response.sources || [];
                    
                    span?.setAttribute('input_sources.count', this.sources.length);
                    span?.setAttribute('sources.loaded', true);
                    span?.addEvent('Input sources loaded successfully');
                    
                    logger.info('Input sources loaded', {
                        'sources.count': this.sources.length,
                    });
                } catch (error) {
                    span?.addEvent('Failed to load input sources');
                    span?.setAttribute('error', true);
                    
                    logger.error('Failed to load input sources', {
                        'error.message': error instanceof Error ? error.message : String(error),
                    });
                    this.sources = [];
                }
            }
        );

        return this.initPromise;
    }

    getSources(): InputSource[] {
        return this.sources;
    }

    getDefaultSource(): InputSource | undefined {
        return this.sources.find(src => src.isDefault);
    }

    getById(id: string): InputSource | undefined {
        return this.sources.find(src => src.id === id);
    }

    isInitialized(): boolean {
        return this.sources.length > 0;
    }

    async listAvailableImages(): Promise<StaticImage[]> {
        return telemetryService.withSpanAsync(
            'InputSourceService.listAvailableImages',
            {
                'http.method': 'POST',
                'rpc.service': 'FileService',
                'rpc.method': 'listAvailableImages',
            },
            async (span) => {
                try {
                    span?.addEvent('Fetching available images');
                    const response = await this.fileClient.listAvailableImages({});
                    
                    const images = response.images || [];
                    
                    span?.setAttribute('available_images.count', images.length);
                    span?.addEvent('Available images loaded successfully');
                    
                    logger.debug('Available images loaded', {
                        'images.count': images.length,
                    });
                    return images;
                } catch (error) {
                    span?.addEvent('Failed to load available images');
                    span?.setAttribute('error', true);
                    
                    logger.error('Failed to load available images', {
                        'error.message': error instanceof Error ? error.message : String(error),
                    });
                    return [];
                }
            }
        );
    }

    async listAvailableVideos(): Promise<StaticVideo[]> {
        return telemetryService.withSpanAsync(
            'InputSourceService.listAvailableVideos',
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
                    
                    span?.setAttribute('available_videos.count', videos.length);
                    span?.addEvent('Available videos loaded successfully');
                    
                    logger.debug('Available videos loaded', {
                        'videos.count': videos.length,
                    });
                    return videos;
                } catch (error) {
                    span?.addEvent('Failed to load available videos');
                    span?.setAttribute('error', true);
                    
                    logger.error('Failed to load available videos', {
                        'error.message': error instanceof Error ? error.message : String(error),
                    });
                    return [];
                }
            }
        );
    }
}

export const inputSourceService = new InputSourceService();

