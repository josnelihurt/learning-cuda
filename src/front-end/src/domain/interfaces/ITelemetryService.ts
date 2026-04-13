import type { Span } from '@opentelemetry/api';

export interface ITelemetryService {
  initialize(): Promise<void>;
  createSpan(name: string, attributes?: Record<string, any>): Span | null;
  withSpan<T>(name: string, attributes: Record<string, any>, fn: (span: Span) => T): T;
  withSpanAsync<T>(name: string, attributes: Record<string, any>, fn: (span: Span | null) => Promise<T>): Promise<T>;
  getTraceHeaders(): Record<string, string>;
  isEnabled(): boolean;
}
