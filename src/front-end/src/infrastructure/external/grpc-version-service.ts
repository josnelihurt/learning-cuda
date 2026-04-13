import { createPromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ImageProcessorService } from '../../gen/image_processor_service_connect';
import { GetVersionInfoRequest } from '../../gen/image_processor_service_pb';
import { TraceContext } from '../../gen/common_pb';
import { telemetryService } from '../observability/telemetry-service';
import { logger } from '../observability/otel-logger';

const tracingInterceptor: Interceptor = (next) => async (req) => {
  const traceHeaders = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(traceHeaders)) {
    req.header.set(key, value);
  }
  return await next(req);
};

class GrpcVersionService {
  private client;

  constructor() {
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      interceptors: [tracingInterceptor],
      useHttpGet: true,
    });
    this.client = createPromiseClient(ImageProcessorService, transport);
  }

  async getVersionInfo() {
    return telemetryService.withSpanAsync(
      'GrpcVersionService.getVersionInfo',
      {
        'rpc.service': 'ImageProcessorService',
        'rpc.method': 'GetVersionInfo',
      },
      async (span) => {
        try {
          span?.addEvent('Creating GetVersionInfo request');

          const traceHeaders = telemetryService.getTraceHeaders();
          const traceContext = traceHeaders['traceparent']
            ? new TraceContext({ traceparent: traceHeaders['traceparent'] })
            : undefined;

          const request = new GetVersionInfoRequest({
            apiVersion: '2.1.0',
            traceContext,
          });

          span?.addEvent('Sending Connect-RPC request');
          const versionInfo = await this.client.getVersionInfo(request);

          if (versionInfo.code !== 0) {
            span?.setAttribute('error', true);
            span?.setAttribute('error.message', versionInfo.message);
            throw new Error(`gRPC error: ${versionInfo.message}`);
          }

          span?.setAttribute('version.server', versionInfo.serverVersion);
          span?.setAttribute('version.library', versionInfo.libraryVersion);
          span?.setAttribute('version.build_date', versionInfo.buildDate);
          span?.setAttribute('version.build_commit', versionInfo.buildCommit);

          logger.info('gRPC version info retrieved', {
            'version.server': versionInfo.serverVersion,
            'version.library': versionInfo.libraryVersion,
            'version.build_date': versionInfo.buildDate,
          });

          return versionInfo;
        } catch (error) {
          span?.setAttribute('error', true);
          span?.setAttribute('error.message', error instanceof Error ? error.message : String(error));
          logger.error('Failed to get gRPC version info', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          throw error;
        }
      }
    );
  }
}

export const grpcVersionService = new GrpcVersionService();

