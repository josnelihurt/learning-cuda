export interface ILogger {
  initialize(logLevel: string, consoleLogging: boolean): void;
  debug(message: string, attributes?: Record<string, any>): void;
  info(message: string, attributes?: Record<string, any>): void;
  warn(message: string, attributes?: Record<string, any>): void;
  error(message: string, attributes?: Record<string, any>): void;
  shutdown(): void;
}
