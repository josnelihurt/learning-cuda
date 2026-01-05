import { SeverityNumber } from '@opentelemetry/api-logs';
import { telemetryService } from './telemetry-service';
import type { ILogger } from '../../domain/interfaces/ILogger';

declare const __APP_VERSION__: string;

const SERVICE_NAME = 'web-app';

type LogAttributes = Record<string, string | number | boolean>;

class OtelLogger implements ILogger {
  private consoleEnabled: boolean = true;
  private minLogLevel: SeverityNumber = SeverityNumber.INFO;
  private initialized = false;
  private logQueue: any[] = [];
  private flushTimer: number | null = null;
  private environment: string = 'development';

  initialize(logLevel: string, consoleEnabled: boolean, environment?: string): void {
    try {
      this.consoleEnabled = consoleEnabled;
      this.minLogLevel = this.parseLogLevel(logLevel);
      if (environment) {
        this.environment = environment;
      }
      this.initialized = true;

      this.flushTimer = window.setInterval(() => {
        this.flushLogs();
      }, 5000);
    } catch (error) {
      console.warn('Failed to initialize logger:', error);
      this.initialized = false;
    }
  }

  debug(message: string, attributes?: LogAttributes): void {
    this.log(SeverityNumber.DEBUG, 'DEBUG', message, attributes);
  }

  info(message: string, attributes?: LogAttributes): void {
    this.log(SeverityNumber.INFO, 'INFO', message, attributes);
  }

  warn(message: string, attributes?: LogAttributes): void {
    this.log(SeverityNumber.WARN, 'WARN', message, attributes);
  }

  error(message: string, attributes?: LogAttributes): void {
    this.log(SeverityNumber.ERROR, 'ERROR', message, attributes);
  }

  isDebugEnabled(): boolean {
    return this.minLogLevel <= SeverityNumber.DEBUG;
  }

  async shutdown(): Promise<void> {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    await this.flushLogs();
  }

  async flush(): Promise<void> {
    await this.flushLogs();
  }

  setEnvironment(environment: string): void {
    this.environment = environment;
  }

  private async flushLogs(): Promise<void> {
    if (this.logQueue.length === 0) {
      return;
    }

    const logsToSend = [...this.logQueue];
    this.logQueue = [];

    const serviceVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : '1.0.0';

    try {
      await fetch(`${window.location.origin}/api/logs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          resourceLogs: [
            {
              resource: {
                attributes: [
                  { key: 'service.name', value: { stringValue: SERVICE_NAME } },
                  { key: 'service.version', value: { stringValue: serviceVersion } },
                  { key: 'environment', value: { stringValue: this.environment } },
                ],
              },
              scopeLogs: [
                {
                  scope: {
                    name: SERVICE_NAME,
                    version: serviceVersion,
                  },
                  logRecords: logsToSend,
                },
              ],
            },
          ],
        }),
      });
    } catch (error) {
      console.warn('Failed to send logs:', error);
    }
  }

  private shouldLog(level: SeverityNumber): boolean {
    return level >= this.minLogLevel;
  }

  private log(
    severityNumber: SeverityNumber,
    severityText: string,
    message: string,
    attributes?: LogAttributes
  ): void {
    if (this.consoleEnabled) {
      this.logToConsole(severityNumber, message, attributes);
    }

    if (!this.initialized || !this.shouldLog(severityNumber)) {
      return;
    }

    const traceHeaders = telemetryService.getTraceHeaders() || {};

    const logAttributes: Record<string, any> = {
      ...attributes,
    };

    if (traceHeaders['traceparent']) {
      logAttributes['trace.traceparent'] = traceHeaders['traceparent'];
    }
    if (traceHeaders['tracestate']) {
      logAttributes['trace.tracestate'] = traceHeaders['tracestate'];
    }

    this.logQueue.push({
      timeUnixNano: Date.now() * 1000000,
      severityNumber,
      severityText,
      body: { stringValue: message },
      attributes: Object.entries(logAttributes).map(([key, value]) => ({
        key,
        value: { stringValue: String(value) },
      })),
    });

    if (this.logQueue.length >= 100) {
      this.flushLogs();
    }
  }

  private logToConsole(
    severityNumber: SeverityNumber,
    message: string,
    attributes?: LogAttributes
  ): void {
    const args = attributes ? [message, attributes] : [message];

    switch (severityNumber) {
      case SeverityNumber.DEBUG:
      case SeverityNumber.DEBUG2:
      case SeverityNumber.DEBUG3:
      case SeverityNumber.DEBUG4:
        console.debug(...args);
        break;
      case SeverityNumber.INFO:
      case SeverityNumber.INFO2:
      case SeverityNumber.INFO3:
      case SeverityNumber.INFO4:
        console.info(...args);
        break;
      case SeverityNumber.WARN:
      case SeverityNumber.WARN2:
      case SeverityNumber.WARN3:
      case SeverityNumber.WARN4:
        console.warn(...args);
        break;
      case SeverityNumber.ERROR:
      case SeverityNumber.ERROR2:
      case SeverityNumber.ERROR3:
      case SeverityNumber.ERROR4:
      case SeverityNumber.FATAL:
      case SeverityNumber.FATAL2:
      case SeverityNumber.FATAL3:
      case SeverityNumber.FATAL4:
        console.error(...args);
        break;
    }
  }

  private parseLogLevel(level: string): SeverityNumber {
    switch (level.toUpperCase()) {
      case 'DEBUG':
        return SeverityNumber.DEBUG;
      case 'INFO':
        return SeverityNumber.INFO;
      case 'WARN':
        return SeverityNumber.WARN;
      case 'ERROR':
        return SeverityNumber.ERROR;
      default:
        return SeverityNumber.INFO;
    }
  }
}

export const logger = new OtelLogger();
