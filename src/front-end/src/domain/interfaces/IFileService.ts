import type { StaticImage } from '../../gen/config_service_pb';

export interface IFileService {
  initialize(): Promise<void>;
  isInitialized(): boolean;
  listAvailableImages(): Promise<StaticImage[]>;
  uploadImage(file: File): Promise<StaticImage>;
}
