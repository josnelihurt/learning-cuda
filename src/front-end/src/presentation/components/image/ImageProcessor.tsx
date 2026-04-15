import { useState, useCallback, useMemo } from 'react';
import { ImageUpload } from './ImageUpload';
import { FilterPanel, type ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import { useImageProcessing } from '@/presentation/hooks/useImageProcessing';
import type { StaticImage } from '@/gen/common_pb';
import styles from './ImageProcessor.module.css';

/**
 * ImageProcessor component
 *
 * Orchestrates the complete image processing workflow:
 * 1. Upload or select an image
 * 2. Configure filters and parameters
 * 3. Process the image via gRPC
 * 4. Display results with original/processed toggle
 *
 * @example
 * ```tsx
 * <ImageProcessor />
 * ```
 */
export function ImageProcessor() {
  // State for selected image from upload or file selector
  const [selectedImage, setSelectedImage] = useState<StaticImage | null>(null);

  // State for active filters from FilterPanel
  const [activeFilters, setActiveFilters] = useState<ActiveFilterState[]>([]);

  // State for toggling between original and processed image view
  const [showOriginal, setShowOriginal] = useState(true);

  // Hook for image processing logic (gRPC calls, progress, errors)
  const { processing, progress, processedImageUrl, error, processImage, reset } =
    useImageProcessing();

  /**
   * Handle image uploaded from ImageUpload component
   * Sets the selected image and clears any previous processed results
   */
  const handleImageUploaded = useCallback(
    (image: StaticImage) => {
      setSelectedImage(image);
      // Clear processed result when new image is uploaded
      reset();
    },
    [reset]
  );

  /**
   * Handle filter changes from FilterPanel component
   * Updates the active filters state
   */
  const handleFiltersChange = useCallback((filters: ActiveFilterState[]) => {
    setActiveFilters(filters);
  }, []);

  /**
   * Handle process button click
   * Validates that image and filters are selected, then triggers processing
   */
  const handleProcessClick = useCallback(async () => {
    if (!selectedImage) {
      return;
    }

    await processImage(selectedImage.path, activeFilters);
  }, [selectedImage, activeFilters, processImage]);

  /**
   * Handle reset button click
   * Clears processed result and error state
   */
  const handleResetClick = useCallback(() => {
    reset();
  }, [reset]);

  /**
   * Computed value: whether processing can be started
   * Requires both an image and at least one active filter
   */
  const canProcess = useMemo(
    () => selectedImage !== null && activeFilters.some((f) => f.id !== 'none'),
    [selectedImage, activeFilters]
  );

  /**
   * Computed value: whether we have a processed result to display
   */
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
          aria-label={processing ? 'Processing image' : 'Process image'}
          title={canProcess ? 'Apply filters to image' : 'Upload an image and select filters first'}
        >
          {processing ? 'Processing...' : 'Process Image'}
        </button>
        <button
          className={styles.resetButton}
          onClick={handleResetClick}
          disabled={!hasProcessedResult}
          data-testid="reset-button"
          aria-label="Reset processed result"
          title={hasProcessedResult ? 'Clear processed result' : 'No result to reset'}
        >
          Reset
        </button>
      </div>

      {/* Middle section: Split layout with upload and filter panels */}
      <div className={styles.contentArea}>
        {/* Left: Image upload */}
        <div className={styles.leftPanel}>
          <div className={styles.panelLabel}>Image Upload</div>
          <ImageUpload onImageUploaded={handleImageUploaded} />

          {/* Display selected image info when available */}
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

      {/* Progress bar: visible during processing */}
      {processing && (
        <div className={styles.progressContainer} data-testid="progress-container" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: `${progress}%` }} />
          </div>
          <div className={styles.progressText}>{progress}%</div>
        </div>
      )}

      {/* Error message: visible when processing fails */}
      {error && (
        <div className={styles.errorMessage} data-testid="error-message" role="alert">
          {error}
        </div>
      )}

      {/* Bottom section: Results area */}
      {hasProcessedResult && selectedImage && (
        <div className={styles.resultsArea} data-testid="results-area">
          <div className={styles.resultsHeader}>
            <h3 className={styles.resultsTitle}>Processing Complete</h3>
            <button
              className={styles.toggleButton}
              onClick={() => setShowOriginal(!showOriginal)}
              data-testid="toggle-button"
              aria-label={showOriginal ? 'Show processed image' : 'Show original image'}
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
