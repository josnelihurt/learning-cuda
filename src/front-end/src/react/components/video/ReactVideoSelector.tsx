import { useEffect, useState } from 'react';
import type { StaticVideo } from '@/gen/common_pb';
import type { InputSource } from '@/gen/config_service_pb';
import { videoService } from '@/infrastructure/data/video-service';
import { logger } from '@/infrastructure/observability/otel-logger';

type ReactVideoSelectorProps = {
  reloadKey: number;
  onVideoSelected: (source: InputSource) => void;
};

export function ReactVideoSelector({ reloadKey, onVideoSelected }: ReactVideoSelectorProps) {
  const [videos, setVideos] = useState<StaticVideo[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loadVideos = async () => {
      setLoading(true);
      setError(null);
      try {
        const availableVideos = await videoService.listAvailableVideos();
        if (cancelled) {
          return;
        }
        setVideos(availableVideos);
        const defaultVideo = availableVideos.find((video) => video.isDefault);
        if (defaultVideo) {
          setSelectedVideoId(defaultVideo.id);
        }
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : String(loadError);
        logger.error('Failed to load videos in React selector', {
          'error.message': message,
        });
        if (!cancelled) {
          setError('Failed to load videos');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    void loadVideos();
    return () => {
      cancelled = true;
    };
  }, [reloadKey]);

  if (loading) {
    return <div className="react-video-loading">Loading videos...</div>;
  }

  if (error) {
    return <div className="react-video-error">{error}</div>;
  }

  if (videos.length === 0) {
    return <div className="react-video-empty">No videos available</div>;
  }

  return (
    <div className="react-video-grid">
      {videos.map((video) => (
        <button
          key={video.id}
          type="button"
          className={`react-video-card ${video.id === selectedVideoId ? 'selected' : ''} ${video.isDefault ? 'default' : ''}`}
          onClick={() => {
            setSelectedVideoId(video.id);
            onVideoSelected({
              id: video.id,
              displayName: video.displayName,
              type: 'video',
              imagePath: '',
              isDefault: video.isDefault,
              videoPath: video.path,
              previewImagePath: video.previewImagePath,
            });
          }}
          data-testid={`video-card-${video.id}`}
        >
          <img
            className="react-video-preview-image"
            src={video.previewImagePath || '/static/img/video-placeholder.png'}
            alt={video.displayName}
            loading="lazy"
            onError={(event) => {
              (event.target as HTMLImageElement).src = '/static/img/video-placeholder.png';
            }}
          />
          <div className="react-video-name">
            {video.displayName}
            {video.isDefault ? <span className="react-video-default-badge">Default</span> : null}
          </div>
        </button>
      ))}
    </div>
  );
}
