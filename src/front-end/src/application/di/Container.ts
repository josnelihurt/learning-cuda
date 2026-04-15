import type { IConfigService } from '../../domain/interfaces/IConfigService';
import type { IVideoService } from '../../domain/interfaces/IVideoService';
import type { IFileService } from '../../domain/interfaces/IFileService';
import type { IInputSourceService } from '../../domain/interfaces/IInputSourceService';
import type { IProcessorCapabilitiesService } from '../../domain/interfaces/IProcessorCapabilitiesService';
import type { ITelemetryService } from '../../domain/interfaces/ITelemetryService';
import type { ILogger } from '../../domain/interfaces/ILogger';
import type { IToolsService } from '../../domain/interfaces/IToolsService';
import type { IWebRTCService } from '../../domain/interfaces/IWebRTCService';

import { streamConfigService } from '../services/config-service';
import { processorCapabilitiesService } from '../services/processor-capabilities-service';
import { videoService } from '../../infrastructure/data/video-service';
import { fileService } from '../../infrastructure/data/file-service';
import { inputSourceService } from '../../infrastructure/data/input-source-service';
import { telemetryService } from '../../infrastructure/observability/telemetry-service';
import { logger } from '../../infrastructure/observability/otel-logger';
import { toolsService } from '../../infrastructure/external/tools-service';
import { webrtcService } from '../../infrastructure/connection/webrtc-service';

// TODO: Decouple singleton pattern - implement factory/builder pattern for service instantiation
class DIContainer {
  private static instance: DIContainer;
  
  private constructor() {}
  
  static getInstance(): DIContainer {
    if (!DIContainer.instance) {
      DIContainer.instance = new DIContainer();
    }
    return DIContainer.instance;
  }

  getConfigService(): IConfigService {
    return streamConfigService;
  }

  getVideoService(): IVideoService {
    return videoService;
  }

  getFileService(): IFileService {
    return fileService;
  }

  getInputSourceService(): IInputSourceService {
    return inputSourceService;
  }

  getProcessorCapabilitiesService(): IProcessorCapabilitiesService {
    return processorCapabilitiesService;
  }

  getTelemetryService(): ITelemetryService {
    return telemetryService;
  }

  getLogger(): ILogger {
    return logger;
  }

  getToolsService(): IToolsService {
    return toolsService;
  }

  getWebRTCService(): IWebRTCService {
    return webrtcService;
  }
}

export const container = DIContainer.getInstance();
