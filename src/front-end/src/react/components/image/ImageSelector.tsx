import { useEffect, useState } from 'react';
import type { StaticImage } from '@/gen/common_pb';
import { useFiles } from '../../hooks/useFiles';
import { FileList } from '../files/FileList';
import styles from './ImageSelector.module.css';

interface ImageSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onImageSelect: (image: StaticImage) => void;
  initialSelectedId?: string;
}

export function ImageSelector({
  isOpen,
  onClose,
  onImageSelect,
  initialSelectedId,
}: ImageSelectorProps) {
  const { images, loading } = useFiles();
  const [selectedImageId, setSelectedImageId] = useState<string | null>(initialSelectedId || null);

  // Update selected ID when initialSelectedId changes
  useEffect(() => {
    setSelectedImageId(initialSelectedId || null);
  }, [initialSelectedId]);

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleImageSelect = (image: StaticImage) => {
    setSelectedImageId(image.id);
    onImageSelect(image);
    onClose();
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div
      className={`${styles.backdrop} ${styles.show}`}
      onClick={handleBackdropClick}
      data-testid="image-selector-backdrop"
    >
      <div
        className={`${styles.modal} ${styles.show}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        data-testid="image-selector-modal"
      >
        <div className={styles.modalHeader}>
          <h2 id="modal-title" className={styles.modalTitle}>
            Select Image
          </h2>
          <button
            className={styles.closeBtn}
            onClick={onClose}
            aria-label="Close modal"
            data-testid="modal-close"
          >
            ×
          </button>
        </div>
        <div className={styles.modalContent}>
          <FileList
            images={images}
            selectedImageId={selectedImageId ?? undefined}
            onImageSelect={handleImageSelect}
            layout="grid"
            loading={loading}
          />
        </div>
      </div>
    </div>
  );
}
