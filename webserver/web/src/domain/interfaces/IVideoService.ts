import type { StaticVideo } from '../../gen/common_pb';

export interface IVideoService {
  initialize(): Promise<void>;
  getVideos(): StaticVideo[];
  getDefaultVideo(): StaticVideo | undefined;
  getById(id: string): StaticVideo | undefined;
  isInitialized(): boolean;
  listAvailableVideos(): Promise<StaticVideo[]>;
  uploadVideo(file: File): Promise<StaticVideo | null>;
}
