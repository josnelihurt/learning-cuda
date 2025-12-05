import { createPromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { RemoteManagementService as RemoteManagementServiceClient } from '../../gen/remote_management_service_connect';
import { 
  StartJetsonNanoRequest, 
  StartJetsonNanoResponse,
  CheckAcceleratorHealthRequest,
  CheckAcceleratorHealthResponse,
  AcceleratorHealthStatus,
  MonitorJetsonNanoRequest,
  MonitorJetsonNanoResponse
} from '../../gen/remote_management_service_pb';
import { logger } from '../observability/otel-logger';
import { telemetryService } from '../observability/telemetry-service';

const tracingInterceptor: Interceptor = (next) => async (req) => {
  const headers = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(headers)) {
    req.header.set(key, value);
  }
  return await next(req);
};

export interface StartJetsonNanoEvent {
  status: string;
  step: string;
  message: string;
}

class RemoteManagementService {
  private client;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      interceptors: [tracingInterceptor],
      useHttpGet: true,
    });
    this.client = createPromiseClient(RemoteManagementServiceClient, transport);
  }

  async startJetsonNano(
    onEvent: (event: StartJetsonNanoEvent) => void,
    onError?: (error: Error) => void
  ): Promise<void> {
    try {
      const request = new StartJetsonNanoRequest({});
      const response = await this.client.startJetsonNano(request);

      onEvent({
        status: response.status?.toString() || 'UNKNOWN',
        step: response.step || '',
        message: response.message || '',
      });
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      logger.error('Failed to start Jetson Nano', {
        'error.message': err.message,
      });
      if (onError) {
        onError(err);
      } else {
        throw err;
      }
    }
  }

  async checkAcceleratorHealth(): Promise<CheckAcceleratorHealthResponse> {
    try {
      const request = new CheckAcceleratorHealthRequest({});
      const response = await this.client.checkAcceleratorHealth(request);
      return response;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      logger.error('Failed to check accelerator health', {
        'error.message': err.message,
      });
      throw err;
    }
  }

  async monitorJetsonNano(
    onData: (data: string) => void,
    onError?: (error: Error) => void
  ): Promise<void> {
    try {
      const request = new MonitorJetsonNanoRequest({});
      const stream = this.client.monitorJetsonNano(request);

      for await (const response of stream) {
        if (response.data) {
          onData(response.data);
        }
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      logger.error('Failed to monitor Jetson Nano', {
        'error.message': err.message,
      });
      if (onError) {
        onError(err);
      } else {
        throw err;
      }
    }
  }
}

export const remoteManagementService = new RemoteManagementService();

