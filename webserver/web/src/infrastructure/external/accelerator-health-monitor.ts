import { remoteManagementService } from './remote-management-service';
import { AcceleratorHealthStatus } from '../../gen/remote_management_service_pb';
import { logger } from '../observability/otel-logger';
import { telemetryService } from '../observability/telemetry-service';

export interface HealthCheckResult {
  isHealthy: boolean;
  message: string;
  serverVersion?: string;
  libraryVersion?: string;
}

class AcceleratorHealthMonitor {
  private pollingInterval: number | null = null;
  private rapidCheckTimeout: number | null = null;
  private readonly HEALTHY_POLL_INTERVAL_MS = 30000;
  private readonly RAPID_POLL_INTERVAL_MS = 2000;
  private readonly RAPID_CHECKS_COUNT = 3;
  private isMonitoring = false;
  private lastHealthStatus: boolean | null = null;
  private onUnhealthyCallback: (() => void) | null = null;
  private rapidCheckCount = 0;
  private isInRapidCheckMode = false;
  private modalOpenCallback: (() => boolean) | null = null;

  startMonitoring(onUnhealthy?: () => void, isModalOpen?: () => boolean): void {
    if (this.isMonitoring) {
      logger.warn('Health monitor is already running');
      return;
    }

    this.onUnhealthyCallback = onUnhealthy || null;
    this.modalOpenCallback = isModalOpen || null;
    this.isMonitoring = true;
    this.lastHealthStatus = null;
    this.rapidCheckCount = 0;
    this.isInRapidCheckMode = false;

    logger.info('Starting accelerator health monitor', {
      'monitor.healthy_poll_interval_ms': this.HEALTHY_POLL_INTERVAL_MS,
      'monitor.rapid_poll_interval_ms': this.RAPID_POLL_INTERVAL_MS,
    });

    this.performInitialCheck();
  }

  stopMonitoring(): void {
    if (!this.isMonitoring) {
      return;
    }

    if (this.pollingInterval !== null) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }

    if (this.rapidCheckTimeout !== null) {
      clearTimeout(this.rapidCheckTimeout);
      this.rapidCheckTimeout = null;
    }

    this.isMonitoring = false;
    logger.info('Stopped accelerator health monitor');
  }

  private async performInitialCheck(): Promise<void> {
    const isHealthy = await this.performHealthCheck();
    
    if (isHealthy) {
      this.lastHealthStatus = true;
      this.startHealthyPolling();
    } else {
      this.startRapidChecks();
    }
  }

  private startHealthyPolling(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }

    if (this.rapidCheckTimeout) {
      clearTimeout(this.rapidCheckTimeout);
      this.rapidCheckTimeout = null;
    }
    
    this.isInRapidCheckMode = false;
    this.rapidCheckCount = 0;
    
    this.pollingInterval = window.setInterval(() => {
      this.performHealthCheck().then((isHealthy) => {
        if (!isHealthy) {
          this.startRapidChecks();
        }
      });
    }, this.HEALTHY_POLL_INTERVAL_MS);
  }

  private startRapidChecks(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }

    if (this.rapidCheckTimeout) {
      clearTimeout(this.rapidCheckTimeout);
      this.rapidCheckTimeout = null;
    }
    
    this.isInRapidCheckMode = true;
    this.rapidCheckCount = 0;
    
    const performRapidCheck = async () => {
      if (!this.isMonitoring) {
        return;
      }

      const isHealthy = await this.performHealthCheck();
      
      if (isHealthy) {
        this.lastHealthStatus = true;
        this.startHealthyPolling();
        return;
      }
      
      this.rapidCheckCount++;
      
      if (this.rapidCheckCount >= this.RAPID_CHECKS_COUNT) {
        if (this.modalOpenCallback && this.modalOpenCallback()) {
          logger.warn('Modal already open, skipping callback');
          this.startHealthyPolling();
          return;
        }
        
        if (this.lastHealthStatus !== false) {
          logger.error('Accelerator failed all rapid health checks, opening modal');
          if (this.onUnhealthyCallback) {
            this.onUnhealthyCallback();
          }
        }
        
        this.lastHealthStatus = false;
        this.startHealthyPolling();
      } else {
        this.rapidCheckTimeout = window.setTimeout(performRapidCheck, this.RAPID_POLL_INTERVAL_MS);
      }
    };
    
    performRapidCheck();
  }

  private async performHealthCheck(): Promise<boolean> {
    if (!this.isMonitoring) {
      return false;
    }

    try {
      return await telemetryService.withSpanAsync(
        'accelerator.health.check',
        {
          'health.check.type': 'accelerator',
        },
        async (span) => {
          try {
            const response = await remoteManagementService.checkAcceleratorHealth();
            const isHealthy = response.status === AcceleratorHealthStatus.HEALTHY;

            span?.setAttribute('health.status', isHealthy ? 'healthy' : 'unhealthy');
            span?.setAttribute('health.server_version', response.serverVersion || 'unknown');
            span?.setAttribute('health.library_version', response.libraryVersion || 'unknown');

            if (isHealthy) {
              if (this.lastHealthStatus === false) {
                logger.info('Accelerator health recovered', {
                  'health.server_version': response.serverVersion,
                  'health.library_version': response.libraryVersion,
                });
              }
            } else {
              logger.warn('Accelerator health check failed', {
                'health.status': response.status,
                'health.message': response.message,
                'health.rapid_check_mode': this.isInRapidCheckMode,
                'health.rapid_check_count': this.rapidCheckCount,
              });
            }
            
            return isHealthy;
          } catch (error) {
            span?.setAttribute('error', true);
            
            const errorMsg = error instanceof Error ? error.message : String(error);
            logger.error('Health check error', {
              'error.message': errorMsg,
              'health.rapid_check_mode': this.isInRapidCheckMode,
              'health.rapid_check_count': this.rapidCheckCount,
            });
            
            return false;
          }
        }
      );
    } catch (error) {
      logger.error('Failed to perform health check', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  }

  isRunning(): boolean {
    return this.isMonitoring;
  }

  getLastHealthStatus(): boolean | null {
    return this.lastHealthStatus;
  }
}

export const acceleratorHealthMonitor = new AcceleratorHealthMonitor();

