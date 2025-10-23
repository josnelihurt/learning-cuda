import { trace, Span, SpanStatusCode, context, propagation } from '@opentelemetry/api';
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { Resource } from '@opentelemetry/resources';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { W3CTraceContextPropagator } from '@opentelemetry/core';
import { logger } from './otel-logger';
import type { ITelemetryService } from '../domain/interfaces/ITelemetryService';

class TelemetryService implements ITelemetryService {
  private enabled: boolean = false;
  private tracer: any = null;

  async initialize() {
    try {
      this.enabled = true;

      if (!this.enabled) {
        logger.info('Telemetry disabled by configuration');
        return;
      }

      const browserInfo = this.getBrowserInfo();

      const resource = new Resource({
        'service.name': 'cuda-image-processor-web',
        'service.version': '1.0.0',
        'browser.name': browserInfo.name,
        'browser.version': browserInfo.version,
        'browser.platform': navigator.platform,
        'browser.language': navigator.language,
        'browser.user_agent': navigator.userAgent,
      });

      const provider = new WebTracerProvider({
        resource: resource,
      });

      const traceUrl = `${window.location.protocol}//${window.location.host}/api/traces`;

      const exporter = new OTLPTraceExporter({
        url: traceUrl,
        headers: {},
      });

      const spanProcessor = new BatchSpanProcessor(exporter);
      provider.addSpanProcessor(spanProcessor);

      const propagator = new W3CTraceContextPropagator();
      provider.register({
        propagator: propagator,
      });

      this.tracer = trace.getTracer('cuda-image-processor-web');

      logger.info('OpenTelemetry initialized for browser');
    } catch (error) {
      logger.warn('Failed to initialize telemetry', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      this.enabled = false;
    }
  }

  createSpan(name: string, attributes?: Record<string, any>): Span | null {
    if (!this.enabled || !this.tracer) {
      return null;
    }

    const span = this.tracer.startSpan(name);

    if (attributes) {
      Object.entries(attributes).forEach(([key, value]) => {
        span.setAttribute(key, value);
      });
    }

    return span;
  }

  withSpan<T>(name: string, attributes: Record<string, any>, fn: (span: Span) => T): T {
    const span = this.createSpan(name, attributes);
    if (!span) {
      return fn(null as any);
    }

    try {
      const result = fn(span);
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error instanceof Error ? error.message : String(error),
      });
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  async withSpanAsync<T>(
    name: string,
    attributes: Record<string, any>,
    fn: (span: Span) => Promise<T>
  ): Promise<T> {
    const span = this.createSpan(name, attributes);
    if (!span) {
      return fn(null as any);
    }

    const ctx = trace.setSpan(context.active(), span);

    try {
      const result = await context.with(ctx, async () => {
        return await fn(span);
      });
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error instanceof Error ? error.message : String(error),
      });
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  shutdown(): Promise<void> {
    if (this.provider) {
      return this.provider.shutdown();
    }
    return Promise.resolve();
  }

  getTraceHeaders(): Record<string, string> {
    if (!this.enabled) {
      return {};
    }

    const headers: Record<string, string> = {};
    const carrier: { [key: string]: string } = {};

    propagation.inject(context.active(), carrier);

    Object.assign(headers, carrier);

    return headers;
  }

  private getBrowserInfo(): { name: string; version: string } {
    const ua = navigator.userAgent;
    let name = 'Unknown';
    let version = 'Unknown';

    if (ua.indexOf('Firefox') > -1) {
      name = 'Firefox';
      const match = ua.match(/Firefox\/(\d+\.\d+)/);
      version = match ? match[1] : 'Unknown';
    } else if (ua.indexOf('Edg') > -1) {
      name = 'Edge';
      const match = ua.match(/Edg\/(\d+\.\d+)/);
      version = match ? match[1] : 'Unknown';
    } else if (ua.indexOf('Chrome') > -1) {
      name = 'Chrome';
      const match = ua.match(/Chrome\/(\d+\.\d+)/);
      version = match ? match[1] : 'Unknown';
    } else if (ua.indexOf('Safari') > -1) {
      name = 'Safari';
      const match = ua.match(/Version\/(\d+\.\d+)/);
      version = match ? match[1] : 'Unknown';
    }

    return { name, version };
  }
}

export const telemetryService = new TelemetryService();
