import { createPromiseClient, PromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService as ConfigServiceClient } from '../gen/config_service_connect';
import { GetSystemInfoResponse } from '../gen/config_service_pb';
import { telemetryService } from './telemetry-service';
import { logger } from './otel-logger';

const tracingInterceptor: Interceptor = (next) => async (req) => {
  const headers = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(headers)) {
    req.header.set(key, value);
  }
  return await next(req);
};

class SystemInfoService {
  private client: PromiseClient<typeof ConfigServiceClient>;
  private systemInfo: GetSystemInfoResponse | null = null;
  private initPromise: Promise<void> | null = null;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      interceptors: [tracingInterceptor],
      useHttpGet: true,
    });

    this.client = createPromiseClient(ConfigServiceClient, transport);
  }

  async initialize(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = telemetryService.withSpanAsync(
      'SystemInfoService.initialize',
      {
        'http.method': 'GET',
        'http.url': window.location.origin,
        'rpc.service': 'ConfigService',
        'rpc.method': 'getSystemInfo',
      },
      async (span) => {
        try {
          span?.addEvent('Fetching system information');
          const response = await this.client.getSystemInfo({});

          this.systemInfo = response;

          span?.setAttribute('system.version.cpp', response.version?.cppVersion || '');
          span?.setAttribute('system.version.go', response.version?.goVersion || '');
          span?.setAttribute('system.version.js', response.version?.jsVersion || '');
          span?.setAttribute('system.version.branch', response.version?.branch || '');
          span?.setAttribute('system.version.commit_hash', response.version?.commitHash || '');
          span?.setAttribute('system.environment', response.environment);
          span?.setAttribute('system.current_library', response.currentLibrary);
          span?.setAttribute('system.api_version', response.apiVersion);
          span?.setAttribute('system.available_libraries_count', response.availableLibraries.length);

          span?.addEvent('System information loaded successfully');

          logger.info('System information loaded', {
            'system.version.js': response.version?.jsVersion || '',
            'system.version.branch': response.version?.branch || '',
            'system.environment': response.environment,
            'system.current_library': response.currentLibrary,
          });
        } catch (error) {
          span?.addEvent('Failed to load system info');
          span?.setAttribute('error', true);

          logger.error('Failed to load system info', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          throw error;
        }
      }
    );

    return this.initPromise;
  }

  async getSystemInfo(): Promise<GetSystemInfoResponse> {
    if (!this.systemInfo) {
      await this.initialize();
    }
    
    if (!this.systemInfo) {
      throw new Error('System info not available');
    }

    return this.systemInfo;
  }

  isInitialized(): boolean {
    return this.systemInfo !== null;
  }

  async refresh(): Promise<void> {
    this.initPromise = null;
    this.systemInfo = null;
    await this.initialize();
  }
}

export const systemInfoService = new SystemInfoService();

