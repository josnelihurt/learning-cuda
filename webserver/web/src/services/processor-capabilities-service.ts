import { createPromiseClient, PromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService as ConfigServiceClient } from '../gen/config_service_connect';
import { ImageProcessorService } from '../gen/image_processor_service_connect';
import { FilterDefinition } from '../gen/common_pb';
import {
  GenericFilterDefinition,
} from '../gen/image_processor_service_pb';
import { telemetryService } from './telemetry-service';
import { logger } from './otel-logger';
import {
  Filter,
  createFilterFromDefinition,
  createFilterFromGenericDefinition,
} from '../components/app/filter-panel.types';
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
  private imageProcessorClient: PromiseClient<typeof ImageProcessorService>;
  private filterDefinitions: FilterDefinition[] = [];
  private filters: Filter[] = [];
  private initPromise: Promise<void> | null = null;
  private genericFilters: GenericFilterDefinition[] = [];
  private genericFiltersLoaded = false;
  private listFiltersPromise: Promise<void> | null = null;
  private filterListeners = new Set<() => void>();

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      interceptors: [tracingInterceptor],
      useHttpGet: true,
    });

    this.client = createPromiseClient(ConfigServiceClient, transport);
    this.imageProcessorClient = createPromiseClient(ImageProcessorService, transport);
  }

  async initialize(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = telemetryService.withSpanAsync(
      'ProcessorCapabilitiesService.initialize',
      {
        'http.method': 'GET',
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
            this.notifyFilterListeners();

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

        await this.fetchGenericFilters();
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

  getGenericFilters(): GenericFilterDefinition[] {
    return this.genericFilters.slice();
  }

  addFiltersUpdatedListener(listener: () => void): void {
    this.filterListeners.add(listener);
  }

  removeFiltersUpdatedListener(listener: () => void): void {
    this.filterListeners.delete(listener);
  }

  private async fetchGenericFilters(): Promise<void> {
    if (this.genericFiltersLoaded) {
      return;
    }

    if (this.listFiltersPromise) {
      return this.listFiltersPromise;
    }

    this.listFiltersPromise = telemetryService.withSpanAsync(
      'ProcessorCapabilitiesService.fetchGenericFilters',
      {
        'rpc.service': 'ImageProcessorService',
        'rpc.method': 'listFilters',
      },
      async (span) => {
        try {
          span?.addEvent('Fetching generic filter definitions');
          const response = await this.imageProcessorClient.listFilters({});
          this.genericFilters = response.filters ?? [];
          this.genericFiltersLoaded = true;

          if (this.genericFilters.length > 0) {
            // Use generic filters if we don't have filters from capabilities, otherwise merge
            if (this.filters.length === 0) {
              this.filters = this.genericFilters.map(createFilterFromGenericDefinition);
            } else {
              // Merge generic filters with existing filters, avoiding duplicates
              const existingIds = new Set(this.filters.map(f => f.id));
              const newFilters = this.genericFilters
                .filter(gf => !existingIds.has(gf.id))
                .map(createFilterFromGenericDefinition);
              this.filters = [...this.filters, ...newFilters];
            }
            this.notifyFilterListeners();
          }

          span?.setAttribute('generic_filters.count', this.genericFilters.length);
          logger.info('Generic filters loaded', {
            'generic_filters.count': this.genericFilters.length,
            'generic_filters.api_version': response.apiVersion,
          });
        } catch (error) {
          span?.setAttribute('error', true);
          logger.warn('Failed to load generic filters', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          this.genericFilters = [];
          this.genericFiltersLoaded = false;
        } finally {
          this.listFiltersPromise = null;
        }
      }
    );

    return this.listFiltersPromise;
  }

  private notifyFilterListeners(): void {
    this.filterListeners.forEach((listener) => {
      try {
        listener();
      } catch (error) {
        logger.warn('Filter listener threw an error', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      }
    });
  }
}


ProcessorCapabilitiesService.prototype['notifyFilterListeners'] = function notifyFilterListeners(this: ProcessorCapabilitiesService) {
  this.filterListeners.forEach((listener) => notifyListener(listener));
};

export const processorCapabilitiesService = new ProcessorCapabilitiesService();
