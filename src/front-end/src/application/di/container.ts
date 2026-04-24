import type { IConfigService } from '@/domain/interfaces/i-config-service';
import type { IVideoService } from '@/domain/interfaces/i-video-service';
import type { IFileService } from '@/domain/interfaces/i-file-service';
import type { IInputSourceService } from '@/domain/interfaces/i-input-source-service';
import type { IProcessorCapabilitiesService } from '@/domain/interfaces/i-processor-capabilities-service';
import type { ITelemetryService } from '@/domain/interfaces/i-telemetry-service';
import type { ILogger } from '@/domain/interfaces/i-logger';
import type { IToolsService } from '@/domain/interfaces/i-tools-service';
import type { IWebRTCService } from '@/domain/interfaces/i-webrtc-service';

import { streamConfigService } from '@/application/services/config-service';
import { processorCapabilitiesService } from '@/application/services/processor-capabilities-service';
import { videoService } from '@/infrastructure/data/video-service';
import { fileService } from '@/infrastructure/data/file-service';
import { inputSourceService } from '@/infrastructure/data/input-source-service';
import { telemetryService } from '@/infrastructure/observability/telemetry-service';
import { logger } from '@/infrastructure/observability/otel-logger';
import { toolsService } from '@/infrastructure/external/tools-service';
import { webrtcService } from '@/infrastructure/connection/webrtc-service';

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
