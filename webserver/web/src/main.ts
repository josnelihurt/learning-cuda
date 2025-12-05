import './components/video/camera-preview';
import './components/app/toast-container';
import './components/app/stats-panel';
import './components/app/filter-panel';
import './components/flags/sync-flags-button';
import './components/ui/tools-dropdown';
import './components/video/video-grid';
import './components/app/source-drawer';
import './components/ui/add-source-fab';
import './components/ui/accelerator-status-fab';
import './components/app/version-footer';
import './components/ui/version-tooltip-lit';
import './components/image/image-selector-modal';
import './components/flags/feature-flags-modal';
import './components/flags/feature-flags-button';
import './components/app/grpc-status-modal';
import './components/video/video-selector';
import './components/video/video-upload';
import './components/app/information-banner';
import './components/app/app-tour';
import './components/app/app-root';
import { acceleratorHealthMonitor } from './infrastructure/external/accelerator-health-monitor';
import { container } from './application/di';
import type {
  IConfigService,
  ITelemetryService,
  ILogger,
  IInputSourceService,
  IProcessorCapabilitiesService,
  IToolsService,
  IVideoService,
  IWebRTCService,
} from './application/di';
import type { AppRoot } from './components/app/app-root';

console.log(`CUDA Image Processor v${__APP_VERSION__} (${__APP_BRANCH__}) - ${__BUILD_TIME__}`);

const streamConfigService: IConfigService = container.getConfigService();
const telemetryService: ITelemetryService = container.getTelemetryService();
const logger: ILogger = container.getLogger();
const inputSourceService: IInputSourceService = container.getInputSourceService();
const processorCapabilitiesService: IProcessorCapabilitiesService = container.getProcessorCapabilitiesService();
const toolsService: IToolsService = container.getToolsService();
const videoService: IVideoService = container.getVideoService();
const webrtcService: IWebRTCService = container.getWebRTCService();

const app = {
  appRoot: null as AppRoot | null,

  async init() {
    const toastManager = document.querySelector('toast-container');
    if (toastManager) {
      toastManager.configure({ duration: 7000 });
    }

    logger.info('Initializing dashboard...');

    const coreServicesResults = await Promise.allSettled([
      telemetryService.initialize(),
      streamConfigService.initialize(),
    ]);

    coreServicesResults.forEach((result, index) => {
      if (result.status === 'rejected') {
        const serviceName = index === 0 ? 'Telemetry' : 'Config';
        logger.error(`${serviceName} service failed to initialize`, {
          'error.message': result.reason instanceof Error ? result.reason.message : String(result.reason),
        });
        toastManager?.error(`${serviceName} Error`, `Failed to initialize ${serviceName.toLowerCase()} service`);
      }
    });

    logger.initialize(streamConfigService.getLogLevel(), streamConfigService.getConsoleLogging());

    const dataServicesResults = await Promise.allSettled([
      inputSourceService.initialize(),
      processorCapabilitiesService.initialize(),
      toolsService.initialize(),
      videoService.initialize(),
      webrtcService.initialize(),
    ]);

    dataServicesResults.forEach((result, index) => {
      if (result.status === 'rejected') {
        const serviceNames = ['Input Source', 'Processor Capabilities', 'Tools', 'Video', 'WebRTC'];
        const serviceName = serviceNames[index];
        logger.error(`${serviceName} service failed to initialize`, {
          'error.message': result.reason instanceof Error ? result.reason.message : String(result.reason),
        });
        toastManager?.warning(`${serviceName} Error`, `Failed to load ${serviceName.toLowerCase()} data`);
      }
    });

    const componentDefinitions = [
      'camera-preview',
      'toast-container',
      'stats-panel',
      'filter-panel',
      'video-grid',
      'source-drawer',
      'add-source-fab',
      'tools-dropdown',
      'image-selector-modal',
      'video-selector',
      'video-upload',
      'information-banner',
      'app-tour',
      'app-root',
      'grpc-status-modal',
      'accelerator-status-fab',
    ];

    await Promise.all(componentDefinitions.map((name) => customElements.whenDefined(name)));

    this.appRoot = document.querySelector('app-root');
    if (this.appRoot) {
      this.appRoot.configService = streamConfigService;
      this.appRoot.telemetryService = telemetryService;
      this.appRoot.logger = logger;
      this.appRoot.inputSourceService = inputSourceService;
      this.appRoot.processorCapabilitiesService = processorCapabilitiesService;
      this.appRoot.toolsService = toolsService;
      this.appRoot.videoService = videoService;
      this.appRoot.webrtcService = webrtcService;

      try {
        await this.appRoot.initialize();
        logger.info('Dashboard initialized successfully');
      } catch (error) {
        logger.error('Failed to initialize app-root', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
        toastManager?.error('Initialization Error', 'Failed to initialize dashboard');
      }

    } else {
      logger.error('app-root element not found in DOM');
      toastManager?.error('Critical Error', 'Dashboard component not found');
    }

    this.startHealthMonitoring();
  },

  startHealthMonitoring() {
    const modalElement = document.querySelector('grpc-status-modal');
    
    acceleratorHealthMonitor.startMonitoring(
      () => {
        logger.warn('Accelerator health check detected unhealthy status, opening modal');
        document.dispatchEvent(new CustomEvent('accelerator-unhealthy'));
      },
      () => {
        return modalElement && (modalElement as any).isModalOpen ? (modalElement as any).isModalOpen() : false;
      }
    );
    logger.info('Accelerator health monitoring started');
  },
};

(window as any).app = app;
(window as any).logger = logger;
(window as any).streamConfigService = streamConfigService;

document.addEventListener('DOMContentLoaded', () => {
  app.init();
});

window.addEventListener('beforeunload', () => {
  const activeSessions = webrtcService.getActiveSessions();
  activeSessions.forEach((session) => {
    webrtcService.stopHeartbeat(session.getId());
    webrtcService.closeSession(session.getId()).catch(() => {
      // Ignore errors during page unload
    });
  });
  acceleratorHealthMonitor.stopMonitoring();
  logger.shutdown();
});
