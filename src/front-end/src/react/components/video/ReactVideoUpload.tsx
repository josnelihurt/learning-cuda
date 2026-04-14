import { useState } from 'react';
import { videoService } from '@/infrastructure/data/video-service';
import { logger } from '@/infrastructure/observability/otel-logger';
import type { StaticVideo } from '@/gen/common_pb';

type ReactVideoUploadProps = {
  onVideoUploaded: (video: StaticVideo) => void;
};

export function ReactVideoUpload({ onVideoUploaded }: ReactVideoUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const uploadVideo = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.mp4')) {
      setError('Only MP4 files are supported');
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError('File size must be less than 100MB');
      return;
    }

    setUploading(true);
    setUploadProgress(30);
    setError(null);
    setSuccess(null);

    try {
      const video = await videoService.uploadVideo(file);
      setUploadProgress(100);
      if (video) {
        setSuccess(`Successfully uploaded: ${video.displayName}`);
        onVideoUploaded(video);
        setTimeout(() => setSuccess(null), 3000);
      }
    } catch (uploadError) {
      const message = uploadError instanceof Error ? uploadError.message : 'Upload failed';
      setError(message);
      logger.error('Video upload failed in React upload', {
        'error.message': message,
      });
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  return (
    <div
      className={`react-video-upload-container ${dragging ? 'dragging' : ''}`}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const file = event.dataTransfer?.files?.[0];
        if (file) {
          void uploadVideo(file);
        }
      }}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      data-testid="video-upload-container"
    >
      <input
        type="file"
        id="react-video-file-input"
        accept=".mp4"
        style={{ display: 'none' }}
        disabled={uploading}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            void uploadVideo(file);
          }
          event.target.value = '';
        }}
      />
      <button
        type="button"
        className="react-video-upload-button"
        disabled={uploading}
        onClick={() => {
          const input = document.getElementById('react-video-file-input') as HTMLInputElement | null;
          input?.click();
        }}
        data-testid="upload-button"
      >
        {uploading ? 'Uploading...' : 'Choose MP4 Video'}
      </button>
      <div className="react-video-upload-info">
        or drag and drop an MP4 file here
        <br />
        <small>Maximum size: 100MB</small>
      </div>

      {uploading ? (
        <div className="react-video-progress-bar">
          <div className="react-video-progress-fill" style={{ width: `${uploadProgress}%` }} />
        </div>
      ) : null}
      {error ? (
        <div className="react-video-message error" data-testid="upload-error">
          {error}
        </div>
      ) : null}
      {success ? (
        <div className="react-video-message success" data-testid="upload-success">
          {success}
        </div>
      ) : null}
    </div>
  );
}
