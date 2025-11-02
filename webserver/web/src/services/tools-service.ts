import { createPromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService } from '../gen/config_service_connect';
import type { ToolCategory } from '../gen/config_service_pb';
import type { IToolsService } from '../domain/interfaces/IToolsService';
import { logger } from './otel-logger';

class ToolsService implements IToolsService {
  private client;
  private categories: ToolCategory[] = [];
  private initialized = false;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      useHttpGet: true,
    });
    this.client = createPromiseClient(ConfigService, transport);
  }

  async initialize(): Promise<void> {
    try {
      const response = await this.client.getAvailableTools({});
      this.categories = response.categories;
      this.initialized = true;
      logger.info('Tools service initialized', {
        'categories.count': this.categories.length,
      });

      const totalTools = this.categories.reduce((sum, cat) => sum + cat.tools.length, 0);
      logger.debug('Total tools', {
        'tools.count': totalTools,
      });
    } catch (error) {
      logger.error('Failed to initialize tools service', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
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
