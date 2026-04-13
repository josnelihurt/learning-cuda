import type { InputSource, StaticImage, StaticVideo } from '../../gen/config_service_pb';

export interface IInputSourceService {
  initialize(): Promise<void>;
  getSources(): InputSource[];
  getDefaultSource(): InputSource | undefined;
  isInitialized(): boolean;
  listAvailableImages(): Promise<StaticImage[]>;
  listAvailableVideos(): Promise<StaticVideo[]>;
}
