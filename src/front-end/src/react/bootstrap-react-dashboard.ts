import { container } from '@/application/di';
import { acceleratorHealthMonitor } from '@/infrastructure/external/accelerator-health-monitor';
import { systemInfoService } from '@/infrastructure/external/system-info-service';

declare global {
  interface Window {
    __reactGrpcStatusModal?: {
      isOpen: () => boolean;
    };
  }
}

let bootstrapPromise: Promise<void> | null = null;
let beforeUnloadAttached = false;

function attachBeforeUnload(): void {
  if (beforeUnloadAttached) {
    return;
  }
  beforeUnloadAttached = true;
  window.addEventListener('beforeunload', () => {
    const webrtcService = container.getWebRTCService();
    const logger = container.getLogger();
    const activeSessions = webrtcService.getActiveSessions();
    activeSessions.forEach((session) => {
      webrtcService.stopHeartbeat(session.getId());
      void webrtcService.closeSession(session.getId()).catch((error) => {
        logger.warn('Failed to close WebRTC session on beforeunload', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      });
    });
    acceleratorHealthMonitor.stopMonitoring();
    container.getLogger().shutdown();
  });
}

async function runBootstrap(): Promise<void> {
  const streamConfigService = container.getConfigService();
  const telemetryService = container.getTelemetryService();
  const logger = container.getLogger();
  const inputSourceService = container.getInputSourceService();
  const processorCapabilitiesService = container.getProcessorCapabilitiesService();
  const toolsService = container.getToolsService();
  const videoService = container.getVideoService();
  const webrtcService = container.getWebRTCService();

  logger.info('Initializing dashboard (React)...');

  const coreServicesResults = await Promise.allSettled([
    telemetryService.initialize(),
    streamConfigService.initialize(),
  ]);

  coreServicesResults.forEach((result, index) => {
    if (result.status === 'rejected') {
      const serviceName = index === 0 ? 'Telemetry' : 'Config';
      logger.error(`${serviceName} service failed to initialize`, {
        'error.message':
          result.reason instanceof Error ? result.reason.message : String(result.reason),
      });
      logger.error(`${serviceName} initialization warning surfaced to user`, {
        'service.name': serviceName,
      });
    }
  });

  logger.initialize(streamConfigService.getLogLevel(), streamConfigService.getConsoleLogging());

  try {
    await systemInfoService.initialize();
    const systemInfo = await systemInfoService.getSystemInfo();
    if (systemInfo.environment) {
      logger.setEnvironment(systemInfo.environment);
    }
  } catch (error) {
    logger.warn('Failed to load system info for environment', {
      'error.message': error instanceof Error ? error.message : String(error),
    });
  }

  const dataServicesResults = await Promise.allSettled([
    inputSourceService.initialize(),
    processorCapabilitiesService.initialize(),
    toolsService.initialize(),
    videoService.initialize(),
    webrtcService.initialize(),
  ]);

  dataServicesResults.forEach((result, index) => {
    if (result.status === 'rejected') {
      const serviceNames = [
        'Input Source',
        'Processor Capabilities',
        'Tools',
        'Video',
        'WebRTC',
      ];
      const serviceName = serviceNames[index];
      logger.error(`${serviceName} service failed to initialize`, {
        'error.message':
          result.reason instanceof Error ? result.reason.message : String(result.reason),
      });
      logger.error(`${serviceName} data loading warning surfaced to user`, {
        'service.name': serviceName,
      });
    }
  });

  acceleratorHealthMonitor.startMonitoring(
    () => {
      logger.warn('Accelerator health check detected unhealthy status, opening modal');
      document.dispatchEvent(new CustomEvent('accelerator-unhealthy'));
    },
    () => Boolean(window.__grpcStatusModal?.isOpen())
  );
  logger.info('Accelerator health monitoring started');

  attachBeforeUnload();
  logger.info('Dashboard (React) services initialized');
}

export function ensureReactDashboardBootstrap(): Promise<void> {
  if (!bootstrapPromise) {
    bootstrapPromise = runBootstrap();
  }
  return bootstrapPromise;
}
