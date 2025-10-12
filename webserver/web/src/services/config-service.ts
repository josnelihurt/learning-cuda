import { createPromiseClient, PromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService as ConfigServiceClient } from '../gen/image_processing_connect';
import { StreamEndpoint } from '../gen/image_processing_pb';

class StreamConfigService {
    private client: PromiseClient<typeof ConfigServiceClient>;
    private config: StreamEndpoint | null = null;
    private initPromise: Promise<void> | null = null;

    constructor() {
        const transport = createConnectTransport({
            baseUrl: window.location.origin,
        });

        this.client = createPromiseClient(ConfigServiceClient, transport);
    }

    async initialize(): Promise<void> {
        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = (async () => {
            try {
                const response = await this.client.getStreamConfig({});
                
                if (response.endpoints && response.endpoints.length > 0) {
                    this.config = response.endpoints[0];
                    console.log('Stream configuration loaded:', {
                        type: this.config.type,
                        endpoint: this.config.endpoint,
                        transportFormat: this.config.transportFormat,
                    });
                } else {
                    console.warn('No stream endpoints configured, using defaults');
                    this.config = new StreamEndpoint({
                        type: 'websocket',
                        endpoint: '/ws',
                        transportFormat: 'json',
                    });
                }
            } catch (error) {
                console.error('Failed to load stream config, using defaults:', error);
                this.config = new StreamEndpoint({
                    type: 'websocket',
                    endpoint: '/ws',
                    transportFormat: 'json',
                });
            }
        })();

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

