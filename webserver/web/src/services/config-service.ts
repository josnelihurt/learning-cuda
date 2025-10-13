import { createPromiseClient, PromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService as ConfigServiceClient } from '../gen/image_processing_connect';
import { StreamEndpoint } from '../gen/image_processing_pb';
import { telemetryService } from './telemetry-service';

const tracingInterceptor: Interceptor = (next) => async (req) => {
    const headers = telemetryService.getTraceHeaders();
    for (const [key, value] of Object.entries(headers)) {
        req.header.set(key, value);
    }
    return await next(req);
};

class StreamConfigService {
    private client: PromiseClient<typeof ConfigServiceClient>;
    private config: StreamEndpoint | null = null;
    private initPromise: Promise<void> | null = null;

    constructor() {
        const transport = createConnectTransport({
            baseUrl: window.location.origin,
            interceptors: [tracingInterceptor],
        });

        this.client = createPromiseClient(ConfigServiceClient, transport);
    }

    async initialize(): Promise<void> {
        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = telemetryService.withSpanAsync(
            'ConfigService.initialize',
            {
                'http.method': 'POST',
                'http.url': window.location.origin,
                'rpc.service': 'ConfigService',
                'rpc.method': 'getStreamConfig',
            },
            async (span) => {
                try {
                    span?.addEvent('Fetching stream configuration');
                    const response = await this.client.getStreamConfig({});
                    
                    if (response.endpoints && response.endpoints.length > 0) {
                        this.config = response.endpoints[0];
                        
                        span?.setAttribute('config.endpoint_count', response.endpoints.length);
                        span?.setAttribute('config.type', this.config.type);
                        span?.setAttribute('config.endpoint', this.config.endpoint);
                        span?.setAttribute('config.transport_format', this.config.transportFormat);
                        
                        span?.addEvent('Stream configuration loaded successfully');
                        
                        console.log('Stream configuration loaded:', {
                            type: this.config.type,
                            endpoint: this.config.endpoint,
                            transportFormat: this.config.transportFormat,
                        });
                    } else {
                        span?.addEvent('No endpoints configured, using defaults');
                        console.warn('No stream endpoints configured, using defaults');
                        this.config = new StreamEndpoint({
                            type: 'websocket',
                            endpoint: '/ws',
                            transportFormat: 'json',
                        });
                    }
                } catch (error) {
                    span?.addEvent('Failed to load stream config, using defaults');
                    span?.setAttribute('error', true);
                    
                    console.error('Failed to load stream config, using defaults:', error);
                    this.config = new StreamEndpoint({
                        type: 'websocket',
                        endpoint: '/ws',
                        transportFormat: 'json',
                    });
                }
            }
        );

        return this.initPromise;
    }

    getTransportFormat(): 'json' | 'binary' {
        if (!this.config) {
            console.warn('Config not initialized, returning default: json');
            return 'json';
        }
        return this.config.transportFormat === 'binary' ? 'binary' : 'json';
    }

    getWebSocketEndpoint(): string {
        if (!this.config) {
            console.warn('Config not initialized, returning default: /ws');
            return '/ws';
        }
        return this.config.endpoint;
    }

    isInitialized(): boolean {
        return this.config !== null;
    }
}

export const streamConfigService = new StreamConfigService();

