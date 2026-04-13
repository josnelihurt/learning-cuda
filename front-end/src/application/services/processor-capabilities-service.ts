import { createPromiseClient, PromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService as ConfigServiceClient } from '../../gen/config_service_connect';
import { ImageProcessorService } from '../../gen/image_processor_service_connect';
import { FilterDefinition } from '../../gen/common_pb';
import {
  GenericFilterDefinition,
} from '../../gen/image_processor_service_pb';
import { telemetryService } from '../../infrastructure/observability/telemetry-service';
import { logger } from '../../infrastructure/observability/otel-logger';
import {
  Filter,
  createFilterFromDefinition,
  createFilterFromGenericDefinition,
} from '../../components/app/filter-panel.types';
import type { IProcessorCapabilitiesService } from '../../domain/interfaces/IProcessorCapabilitiesService';

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

    this.initPromise = this.fetchGenericFilters();
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
          span?.addEvent('Fetching filter definitions');
          const response = await this.imageProcessorClient.listFilters({});
          this.genericFilters = response.filters ?? [];
          this.genericFiltersLoaded = true;

          logger.debug('Raw filter response', {
            'filters.count': this.genericFilters.length,
          });

          if (this.genericFilters.length > 0) {
            this.filters = this.genericFilters.map(createFilterFromGenericDefinition);
            logger.debug('Filters mapped from generic definitions', {
              'filters.count': this.filters.length,
            });
            this.filters.forEach((f) => {
              f.parameters.forEach((p) => {
                if (p.options && p.options.length > 0) {
                  logger.debug(`Filter ${f.id} parameter ${p.id} has ${p.options.length} options`, {
                    'filter.id': f.id,
                    'param.id': p.id,
                    'param.name': p.name,
                    'options.count': p.options.length,
                  });
                } else {
                  logger.warn(`Filter ${f.id} parameter ${p.id} has no options`, {
                    'filter.id': f.id,
                    'param.id': p.id,
                    'param.name': p.name,
                    'param.type': p.type,
                  });
                }
              });
            });
            this.notifyFilterListeners();
          }

          span?.setAttribute('filters.count', this.genericFilters.length);
          logger.info('Filters loaded', {
            'filters.count': this.genericFilters.length,
            'filters.api_version': response.apiVersion,
          });
        } catch (error) {
          span?.setAttribute('error', true);
          logger.error('Failed to load filters', {
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

export const processorCapabilitiesService = new ProcessorCapabilitiesService();
