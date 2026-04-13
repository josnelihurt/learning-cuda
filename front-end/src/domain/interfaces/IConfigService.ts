export interface IConfigService {
  initialize(): Promise<void>;
  getTransportFormat(): 'json' | 'binary';
  getWebSocketEndpoint(): string;
  isInitialized(): boolean;
  getLogLevel(): string;
  getConsoleLogging(): boolean;
  setLogLevel(level: string): void;
  setConsoleLogging(enabled: boolean): void;
}
