import { type ReactElement } from 'react';
import { CameraPreview } from './CameraPreview';
import { VideoSourceCard } from './VideoSourceCard';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import styles from './VideoGrid.module.css';

export type GridSource = {
  id: string;
  number: number;
  name: string;
  type: string;
  imageSrc: string;
  remoteStream?: MediaStream | null;
  filters?: ActiveFilterState[];
};

type CameraFramePayload = {
  base64data: string;
  width: number;
  height: number;
  timestamp: number;
};

type VideoGridProps = {
  sources: GridSource[];
  selectedSourceId: string | null;
  onSelectSource: (sourceId: string) => void;
  onCloseSource: (sourceId: string) => void;
  onChangeImageRequest: (sourceId: string, sourceNumber: number) => void;
  onCameraFrame: (sourceId: string, payload: CameraFramePayload) => void;
  onCameraStreamReady: (sourceId: string, stream: MediaStream) => void;
  onCameraStatus: (status: string, type: 'success' | 'error' | 'warning' | 'inactive') => void;
  onCameraError: (title: string, message: string) => void;
};

function getGridClass(count: number): string {
  if (count <= 1) return styles.grid1;
  if (count === 2) return styles.grid2;
  if (count <= 4) return styles.grid4;
  if (count <= 6) return styles.grid6;
  return styles.grid9;
}

export function VideoGrid({
  sources,
  selectedSourceId,
  onSelectSource,
  onCloseSource,
  onChangeImageRequest,
  onCameraFrame,
  onCameraStreamReady,
  onCameraStatus,
  onCameraError,
}: VideoGridProps): ReactElement {
  return (
    <div className={styles.shell}>
      <div
        data-testid="video-grid"
        className={`${styles.grid} ${getGridClass(sources.length)}`}
      >
        {sources.map((source) => (
          <VideoSourceCard
            key={source.id}
            sourceId={source.id}
            sourceNumber={source.number}
            sourceName={source.name}
            sourceType={source.type}
            imageSrc={source.imageSrc}
            isSelected={selectedSourceId === source.id}
            onSelect={onSelectSource}
            onClose={onCloseSource}
            onChangeImage={onChangeImageRequest}
          >
            {source.type === 'camera' ? (
              <CameraPreview
                captureFrames={false}
                remoteStream={source.remoteStream ?? null}
                activeFilters={source.filters ?? []}
                onFrameCaptured={(payload) => onCameraFrame(source.id, payload)}
                onStreamReady={(stream) => onCameraStreamReady(source.id, stream)}
                onCameraStatus={onCameraStatus}
                onCameraError={onCameraError}
              />
            ) : null}
          </VideoSourceCard>
        ))}
      </div>
    </div>
  );
}
