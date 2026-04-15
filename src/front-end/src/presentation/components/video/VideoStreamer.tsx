import { useState, useCallback } from 'react';
import { useWebRTCStream } from '../../hooks/useWebRTCStream';
import { VideoSourceSelector } from './VideoSourceSelector';
import { VideoCanvas } from './VideoCanvas';
import type { StaticImage } from '@/gen/common_pb';
import { useDashboardState } from '../../context/dashboard-state-context';
import styles from './VideoStreamer.module.css';

export function VideoStreamer() {
  const {
    connectionState,
    isStreaming,
    error,
    startStream,
    stopStream,
  } = useWebRTCStream();

  const { activeFilters } = useDashboardState();

  const [selectedSource, setSelectedSource] = useState<{ type: 'camera' | 'file'; id?: string }>({
    type: 'camera',
  });
  const [availableVideos] = useState<StaticImage[]>([]);

  const handleSourceChange = useCallback((source: { type: 'camera' | 'file'; id?: string }) => {
    setSelectedSource(source);
  }, []);

  const handleStartStream = useCallback(async () => {
    const sourceId = selectedSource.type === 'camera' ? 'camera-1' : selectedSource.id || '';
    await startStream(sourceId, activeFilters);
  }, [selectedSource, activeFilters, startStream]);

  const handleStopStream = useCallback(async () => {
    await stopStream();
  }, [stopStream]);

  const isDisabled = connectionState === 'connecting';
  const isLoading = connectionState === 'connecting';
  const hasError = connectionState === 'failed';

  return (
    <div className={styles.streamer} data-testid="video-streamer">
      <div className={styles.controls}>
        <div className={styles.sourceSection}>
          <h3 className={styles.heading}>Video Source</h3>
          <VideoSourceSelector
            availableVideos={availableVideos}
            selectedVideoId={selectedSource.type === 'file' ? selectedSource.id : undefined}
            onSourceChange={handleSourceChange}
            data-testid="video-source-selector"
          />
        </div>

        <div className={styles.actions}>
          {!isStreaming ? (
            <button
              className={styles.startButton}
              onClick={handleStartStream}
              disabled={isDisabled}
              type="button"
            >
              {isLoading ? 'Connecting...' : 'Start Stream'}
            </button>
          ) : (
            <button
              className={styles.stopButton}
              onClick={handleStopStream}
              type="button"
            >
              Stop Stream
            </button>
          )}
        </div>

        {hasError && error && (
          <div className={styles.error}>
            {error.message}
          </div>
        )}
      </div>

      <div className={styles.canvas}>
        {isStreaming ? (
          <VideoCanvas width={640} height={480} />
        ) : (
          <div className={styles.emptyState} data-testid="empty-state">
            <h4>No Stream Active</h4>
            <p>Select a video source to begin streaming. Choose from camera or upload a video file.</p>
          </div>
        )}
      </div>
    </div>
  );
}
