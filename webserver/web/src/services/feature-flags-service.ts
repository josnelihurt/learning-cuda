import { createPromiseClient, PromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { FeatureFlagsService } from '../gen/feature_flags_connect';
import { GetFeatureFlagsRequest } from '../gen/feature_flags_pb';
import { telemetryService } from './telemetry-service';
import { logger } from './otel-logger';

type FeatureFlagValue = string | boolean | null;

class FeatureFlagsServiceClient {
  private client: PromiseClient<typeof FeatureFlagsService>;
  private flagsCache: Map<string, FeatureFlagValue> = new Map();
  private cacheTimestamp: number = 0;
  private readonly CACHE_TTL = 30000;
  private initPromise: Promise<void> | null = null;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
    });
    this.client = createPromiseClient(FeatureFlagsService, transport);
  }

  async initialize(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = telemetryService.withSpanAsync(
      'FeatureFlagsService.initialize',
      {
        'service.name': 'FeatureFlagsService',
        'service.method': 'initialize',
      },
      async (span) => {
        try {
          await this.refreshFlags();
          span?.setAttribute('flags.count', this.flagsCache.size);
          logger.info('Feature flags service initialized', {
            'flags.count': this.flagsCache.size,
          });
        } catch (error) {
          span?.setAttribute('error', true);
          span?.setAttribute('error.message', error instanceof Error ? error.message : String(error));
          logger.error('Failed to initialize feature flags service', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
        }
      }
    );

    return this.initPromise;
  }

  private async refreshFlags(): Promise<void> {
    const now = Date.now();
    if (now - this.cacheTimestamp < this.CACHE_TTL && this.flagsCache.size > 0) {
      return;
    }

    try {
      const response = await this.client.getFeatureFlags({});
      this.flagsCache.clear();

      for (const flag of response.flags) {
        if (flag.type === 'boolean') {
          this.flagsCache.set(flag.key, flag.value === 'true' || flag.value === '1');
        } else {
          this.flagsCache.set(flag.key, flag.value);
        }
      }

      this.cacheTimestamp = now;
    } catch (error) {
      logger.error('Failed to fetch feature flags', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  async getFeatureFlagValue(key: string): Promise<FeatureFlagValue> {
    await this.initialize();
    await this.refreshFlags();
    return this.flagsCache.get(key) ?? null;
  }

  async isFeatureEnabled(key: string): Promise<boolean> {
    const value = await this.getFeatureFlagValue(key);
    if (typeof value === 'boolean') {
      return value;
    }
    if (typeof value === 'string') {
      return value === 'true' || value === '1';
    }
    return false;
  }

  clearCache(): void {
    this.flagsCache.clear();
    this.cacheTimestamp = 0;
  }
}

export const featureFlagsService = new FeatureFlagsServiceClient();

