import { createConnectTransport } from '@connectrpc/connect-web';
import { tracingInterceptor } from './tracing-interceptor';

export function createGrpcConnectTransport() {
  return createConnectTransport({
    baseUrl: window.location.origin,
    interceptors: [tracingInterceptor],
    useHttpGet: true,
  });
}
