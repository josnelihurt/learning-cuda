import type { ToolCategory } from '../../gen/config_service_pb';

export interface IToolsService {
  initialize(): Promise<void>;
  getCategories(): ToolCategory[];
  isInitialized(): boolean;
}
