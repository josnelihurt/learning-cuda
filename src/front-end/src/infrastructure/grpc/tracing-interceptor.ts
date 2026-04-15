import type { Interceptor } from '@connectrpc/connect';
import { telemetryService } from '@/infrastructure/observability/telemetry-service';

export const tracingInterceptor: Interceptor = (next) => async (req) => {
  const headers = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(headers)) {
    req.header.set(key, value);
  }
  return await next(req);
};
