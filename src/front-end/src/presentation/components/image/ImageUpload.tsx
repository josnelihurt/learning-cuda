import { useRef, useState } from 'react';
import { useImageUpload } from '@/presentation/hooks/useImageUpload';
import type { StaticImage } from '@/gen/config_service_pb';
import styles from './ImageUpload.module.css';

interface ImageUploadProps {
  onImageUploaded: (image: StaticImage) => void;
}

export function ImageUpload({ onImageUploaded }: ImageUploadProps) {
  const { uploading, progress, error, uploadFile } = useImageUpload();
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    if (!uploading && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!uploading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (uploading) {
      return;
    }

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const image = await uploadFile(files[0]);
      if (image) {
        onImageUploaded(image);
      }
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const image = await uploadFile(files[0]);
      if (image) {
        onImageUploaded(image);
      }
    }
    // Reset input so same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const containerClasses = [
    styles.uploadContainer,
    isDragging ? styles.dragging : '',
    uploading ? styles.uploading : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div>
      <div
        className={containerClasses}
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        data-testid="upload-container"
      >
        <div className={styles.uploadIcon}>+</div>
        <div className={styles.uploadText}>{uploading ? 'Uploading...' : 'Add Image'}</div>
        <div className={styles.uploadHint}>Click or drag and drop to upload</div>
        <div className={styles.uploadFormat}>Only PNG files supported (max 10MB)</div>

        {uploading && (
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: `${progress}%` }} />
          </div>
        )}
      </div>

      {error && <div className={styles.error} data-testid="upload-error">{error}</div>}

      <input
        type="file"
        ref={fileInputRef}
        accept=".png"
        onChange={handleFileSelect}
        className={styles.fileInput}
        data-testid="file-input"
      />
    </div>
  );
}
