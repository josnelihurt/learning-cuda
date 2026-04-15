/**
 * Interface for components that display transport status statistics.
 * Extracted from Lit StatsPanel during React migration.
 */
export interface IStatsDisplay {
  updateTransportStatus(status: 'connected' | 'disconnected' | 'connecting', text: string): void;
  updateCameraStatus(status: string, type: 'success' | 'error' | 'warning' | 'inactive'): void;
  updateProcessingStats(processingTime: number): void;
}

/**
 * Interface for components that display toast notifications.
 * Extracted from Lit ToastContainer during React migration.
 */
export interface IToastDisplay {
  error(title: string, message: string): string;
  warning(title: string, message: string): string;
  info(title: string, message: string): string;
  success(title: string, message: string): string;
}

/**
 * Interface for components that display camera preview status.
 * Extracted from Lit CameraPreview during React migration.
 */
export interface ICameraPreview {
  setProcessing(isProcessing: boolean): void;
  getLastFrameTime(): number;
}
