import { useState, type ReactElement } from 'react';
import { FileList } from '@/presentation/components/files/FileList';
import type { StaticImage } from '@/gen/common_pb';
import styles from './VideoSourceSelector.module.css';

type SourceType = 'camera' | 'file';

interface VideoSourceSelectorProps {
  availableVideos: StaticImage[];
  selectedVideoId?: string;
  onSourceChange: (source: { type: SourceType; id?: string }) => void;
  className?: string;
}

export function VideoSourceSelector({
  availableVideos,
  selectedVideoId,
  onSourceChange,
  className,
}: VideoSourceSelectorProps): ReactElement {
  const [sourceType, setSourceType] = useState<SourceType>('camera');

  const handleCameraSelect = () => {
    setSourceType('camera');
    onSourceChange({ type: 'camera' });
  };

  const handleFileSelect = () => {
    setSourceType('file');
  };

  const handleVideoSelect = (video: StaticImage) => {
    onSourceChange({ type: 'file', id: video.id });
  };

  return (
    <div className={`${styles.selector} ${className || ''}`} data-testid="video-source-selector">
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${sourceType === 'camera' ? styles.tabActive : ''}`}
          onClick={handleCameraSelect}
          type="button"
          data-testid="camera-tab"
        >
          Camera
        </button>
        <button
          className={`${styles.tab} ${sourceType === 'file' ? styles.tabActive : ''}`}
          onClick={handleFileSelect}
          type="button"
          data-testid="file-tab"
        >
          File
        </button>
      </div>

      {sourceType === 'file' && (
        <div className={styles.fileList}>
          <FileList
            images={availableVideos}
            selectedImageId={selectedVideoId}
            onImageSelect={handleVideoSelect}
            layout="grid"
          />
        </div>
      )}
    </div>
  );
}
