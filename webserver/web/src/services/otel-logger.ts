import { SeverityNumber } from '@opentelemetry/api-logs';
import { LoggerProvider, BatchLogRecordProcessor } from '@opentelemetry/sdk-logs';
import { OTLPLogExporter } from '@opentelemetry/exporter-logs-otlp-http';
import { Resource } from '@opentelemetry/resources';
import { telemetryService } from './telemetry-service';

type LogAttributes = Record<string, string | number | boolean>;

class OtelLogger {
    private logger: any = null;
    private loggerProvider: LoggerProvider | null = null;
    private consoleEnabled: boolean = true;
    private minLogLevel: SeverityNumber = SeverityNumber.INFO;
    private initialized = false;

    initialize(logLevel: string, consoleEnabled: boolean): void {
        this.consoleEnabled = consoleEnabled;
        this.minLogLevel = this.parseLogLevel(logLevel);
        
        const resource = new Resource({
            'service.name': 'cuda-image-processor-web',
            'service.version': '1.0.0',
        });

        this.loggerProvider = new LoggerProvider({ resource });

        const logExporter = new OTLPLogExporter({
            url: `${window.location.origin}/api/logs`,
            headers: {},
        });

        this.loggerProvider.addLogRecordProcessor(
            new BatchLogRecordProcessor(logExporter, {
                maxQueueSize: 100,
                scheduledDelayMillis: 5000,
                maxExportBatchSize: 100,
            })
        );

        this.logger = this.loggerProvider.getLogger('frontend-logger');
        this.initialized = true;
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
        if (this.loggerProvider) {
            await this.loggerProvider.shutdown();
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
        if (!this.initialized || !this.shouldLog(severityNumber)) {
            return;
        }

        if (this.consoleEnabled) {
            this.logToConsole(severityNumber, message, attributes);
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

        this.logger?.emit({
            severityNumber,
            severityText,
            body: message,
            attributes: logAttributes,
        });
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

