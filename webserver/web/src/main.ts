import './components/camera-preview';
import './components/toast-container';
import './components/stats-panel';
import './components/filter-panel';
import './components/sync-flags-button';
import './components/tools-dropdown';
import './components/video-grid';
import './components/source-drawer';
import './components/add-source-fab';
import './components/version-footer';
import './components/version-tooltip-lit';
import './components/image-selector-modal';
import './components/feature-flags-modal';
import './components/feature-flags-button';
import './components/video-selector';
import './components/video-upload';
import './components/information-banner';
import './components/app-tour';
import './components/app-root';
import { container } from './application/di';
import type {
  IConfigService,
  ITelemetryService,
  ILogger,
  IInputSourceService,
  IProcessorCapabilitiesService,
  IToolsService,
  IVideoService,
} from './application/di';
import type { AppRoot } from './components/app-root';

console.log(`CUDA Image Processor v${__APP_VERSION__} (${__APP_BRANCH__}) - ${__BUILD_TIME__}`);

const streamConfigService: IConfigService = container.getConfigService();
const telemetryService: ITelemetryService = container.getTelemetryService();
const logger: ILogger = container.getLogger();
const inputSourceService: IInputSourceService = container.getInputSourceService();
const processorCapabilitiesService: IProcessorCapabilitiesService = container.getProcessorCapabilitiesService();
const toolsService: IToolsService = container.getToolsService();
const videoService: IVideoService = container.getVideoService();

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
    ]);

    dataServicesResults.forEach((result, index) => {
      if (result.status === 'rejected') {
        const serviceNames = ['Input Source', 'Processor Capabilities', 'Tools', 'Video'];
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
  },
};

(window as any).app = app;
(window as any).logger = logger;
(window as any).streamConfigService = streamConfigService;

document.addEventListener('DOMContentLoaded', () => {
  app.init();
});

window.addEventListener('beforeunload', () => {
  logger.shutdown();
});
