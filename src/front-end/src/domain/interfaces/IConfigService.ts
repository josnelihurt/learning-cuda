export interface IConfigService {
  initialize(): Promise<void>;
  isInitialized(): boolean;
  getLogLevel(): string;
  getConsoleLogging(): boolean;
  setLogLevel(level: string): void;
  setConsoleLogging(enabled: boolean): void;
}
