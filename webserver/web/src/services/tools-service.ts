import { createPromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService } from '../gen/config_service_connect';
import type { ToolCategory } from '../gen/config_service_pb';

class ToolsService {
    private client;
    private categories: ToolCategory[] = [];
    private initialized = false;

    constructor() {
        const transport = createConnectTransport({
            baseUrl: window.location.origin,
        });
        this.client = createPromiseClient(ConfigService, transport);
    }

    async initialize(): Promise<void> {
        try {
            const response = await this.client.getAvailableTools({});
            this.categories = response.categories;
            this.initialized = true;
            console.log(`Tools service initialized: ${this.categories.length} categories loaded`);
            
            const totalTools = this.categories.reduce((sum, cat) => sum + cat.tools.length, 0);
            console.log(`Total tools: ${totalTools}`);
        } catch (error) {
            console.error('Failed to initialize tools service:', error);
            this.initialized = false;
        }
    }

    getCategories(): ToolCategory[] {
        return this.categories;
    }

    isInitialized(): boolean {
        return this.initialized;
    }
}

export const toolsService = new ToolsService();

