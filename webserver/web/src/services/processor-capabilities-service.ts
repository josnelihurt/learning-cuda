import { createPromiseClient, PromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService as ConfigServiceClient } from '../gen/config_service_connect';
import { FilterDefinition } from '../gen/common_pb';
import { telemetryService } from './telemetry-service';
import { Filter, createFilterFromDefinition } from '../components/filter-panel.types';
import type { IProcessorCapabilitiesService } from '../domain/interfaces/IProcessorCapabilitiesService';

const tracingInterceptor: Interceptor = (next) => async (req) => {
  const headers = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(headers)) {
    req.header.set(key, value);
  }
  return await next(req);
};

class ProcessorCapabilitiesService implements IProcessorCapabilitiesService {
  private client: PromiseClient<typeof ConfigServiceClient>;
  private filterDefinitions: FilterDefinition[] = [];
  private filters: Filter[] = [];
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
      'ProcessorCapabilitiesService.initialize',
      {
        'http.method': 'POST',
        'http.url': window.location.origin,
        'rpc.service': 'ConfigService',
        'rpc.method': 'getProcessorStatus',
      },
      async (span) => {
        try {
          span?.addEvent('Fetching processor capabilities');
          const response = await this.client.getProcessorStatus({});

          if (response.capabilities && response.capabilities.filters) {
            this.filterDefinitions = response.capabilities.filters;
            this.filters = this.filterDefinitions.map(createFilterFromDefinition);

            span?.setAttribute('capabilities.filter_count', this.filterDefinitions.length);
            span?.setAttribute('capabilities.api_version', response.capabilities.apiVersion);
            span?.addEvent('Processor capabilities loaded successfully');

            logger.info('Processor capabilities loaded', {
              'capabilities.api_version': response.capabilities.apiVersion,
              'capabilities.library_version': response.capabilities.libraryVersion,
              'capabilities.filter_count': this.filterDefinitions.length,
              'capabilities.filters': this.filters.map((f) => f.name).join(','),
            });
          } else {
            span?.addEvent('No capabilities in response');
            logger.warn('No capabilities in response');
          }
        } catch (error) {
          span?.addEvent('Failed to load processor capabilities');
          span?.setAttribute('error', true);
          logger.error('Failed to load processor capabilities', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
        }
      }
    );

    return this.initPromise;
  }

  getFilters(): Filter[] {
    return this.filters.map((f) => ({ ...f }));
  }

  getFilterDefinitions(): FilterDefinition[] {
    return this.filterDefinitions;
  }

  isInitialized(): boolean {
    return this.filters.length > 0;
  }
}

export const processorCapabilitiesService = new ProcessorCapabilitiesService();
