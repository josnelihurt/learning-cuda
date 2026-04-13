import { useState, useCallback } from 'react';
import { ImageUpload } from './ImageUpload';
import { FilterPanel, type ActiveFilterState } from '../filters/FilterPanel';
import { useImageProcessing } from '../../hooks/useImageProcessing';
import type { StaticImage } from '@/gen/common_pb';
import styles from './ImageProcessor.module.css';

export function ImageProcessor() {
  const [selectedImage, setSelectedImage] = useState<StaticImage | null>(null);
  const [activeFilters, setActiveFilters] = useState<ActiveFilterState[]>([]);
  const [showOriginal, setShowOriginal] = useState(true);

  const { processing, progress, processedImageUrl, error, processImage, reset } = useImageProcessing();

  const handleImageUploaded = useCallback((image: StaticImage) => {
    setSelectedImage(image);
    // Clear processed result when new image is uploaded
    reset();
  }, [reset]);

  const handleFiltersChange = useCallback((filters: ActiveFilterState[]) => {
    setActiveFilters(filters);
  }, []);

  const handleProcessClick = useCallback(async () => {
    if (!selectedImage) {
      return;
    }

    await processImage(selectedImage.path, activeFilters);
  }, [selectedImage, activeFilters, processImage]);

  const handleResetClick = useCallback(() => {
    reset();
  }, [reset]);

  const canProcess = selectedImage !== null && activeFilters.some((f) => f.id !== 'none');

  const hasProcessedResult = processedImageUrl !== null;

  return (
    <div className={styles.processorContainer} data-testid="image-processor">
      {/* Top section: Action buttons */}
      <div className={styles.actionBar}>
        <button
          className={styles.processButton}
          onClick={handleProcessClick}
          disabled={!canProcess || processing}
          data-testid="process-button"
        >
          {processing ? 'Processing...' : 'Process Image'}
        </button>
        <button
          className={styles.resetButton}
          onClick={handleResetClick}
          disabled={!hasProcessedResult}
          data-testid="reset-button"
        >
          Reset
        </button>
      </div>

      {/* Middle section: Split layout */}
      <div className={styles.contentArea}>
        {/* Left: Image upload */}
        <div className={styles.leftPanel}>
          <div className={styles.panelLabel}>Image Upload</div>
          <ImageUpload onImageUploaded={handleImageUploaded} />

          {selectedImage && (
            <div className={styles.selectedImageInfo} data-testid="selected-image-info">
              <div className={styles.imageName}>{selectedImage.displayName}</div>
              <div className={styles.imagePath}>{selectedImage.path}</div>
            </div>
          )}
        </div>

        {/* Right: Filter panel */}
        <div className={styles.rightPanel}>
          <div className={styles.panelLabel}>Filters</div>
          <FilterPanel
            onFiltersChange={handleFiltersChange}
            initialActiveFilters={hasProcessedResult ? activeFilters : undefined}
          />
        </div>
      </div>

      {/* Progress bar */}
      {processing && (
        <div className={styles.progressContainer} data-testid="progress-container">
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: `${progress}%` }} />
          </div>
          <div className={styles.progressText}>{progress}%</div>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className={styles.errorMessage} data-testid="error-message">
          {error}
        </div>
      )}

      {/* Bottom section: Results */}
      {hasProcessedResult && selectedImage && (
        <div className={styles.resultsArea} data-testid="results-area">
          <div className={styles.resultsHeader}>
            <h3 className={styles.resultsTitle}>Processing Complete</h3>
            <button
              className={styles.toggleButton}
              onClick={() => setShowOriginal(!showOriginal)}
              data-testid="toggle-button"
            >
              {showOriginal ? 'Show Processed' : 'Show Original'}
            </button>
          </div>

          <div className={styles.imageComparison}>
            {showOriginal ? (
              <div className={styles.imageView}>
                <div className={styles.imageLabel}>Original</div>
                <img
                  src={selectedImage.path}
                  alt="Original"
                  className={styles.imageElement}
                  data-testid="original-image"
                />
              </div>
            ) : (
              <div className={styles.imageView}>
                <div className={styles.imageLabel}>Processed</div>
                <img
                  src={processedImageUrl!}
                  alt="Processed"
                  className={styles.imageElement}
                  data-testid="processed-image"
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
