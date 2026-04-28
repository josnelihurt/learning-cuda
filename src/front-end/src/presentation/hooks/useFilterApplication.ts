import { useCallback, useRef } from 'react';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import type { IToastDisplay } from '@/infrastructure/transport/transport-types';
import { FilterData, GrayscaleAlgorithm } from '@/domain/value-objects';
import { logger } from '@/infrastructure/observability/otel-logger';
import { frameResponseToDataUrl, rasterizeImageToRgb } from '@/presentation/utils/image-utils';
import type { GridSource } from '@/presentation/utils/grid-source';
import { AcceleratorType } from '@/gen/common_pb';

export interface FilterApplicationOptions {
  source: GridSource;
  filters: ActiveFilterState[];
  accelerator: AcceleratorType;
  resolution: string;
  onSourceUpdate: (sourceId: string, updater: (current: GridSource) => GridSource) => void;
}

export type FilterApplicationResult =
  | { status: 'success' }
  | { status: 'error'; message: string };

export function useFilterApplication(toastManager: IToastDisplay): { applyStaticFilters: (options: FilterApplicationOptions) => Promise<FilterApplicationResult>; applyVideoFilters: (options: FilterApplicationOptions) => Promise<FilterApplicationResult> } {
  const pendingUpdateRef = useRef<{ sourceId: string; filters: ActiveFilterState[] } | null>(null);

  const mapFiltersToValueObjects = useCallback((filters: ActiveFilterState[]): FilterData[] => {
    return filters.map((filter) => new FilterData(filter.id, { ...filter.parameters }));
  }, []);

  const applyStaticFilters = useCallback(
    async (options: FilterApplicationOptions): Promise<FilterApplicationResult> => {
      const { source, filters, accelerator, resolution, onSourceUpdate } = options;

      if (!source.transport) {
        logger.error('Static source is missing a WebRTC transport', {
          'source.id': source.id,
        });
        return { status: 'error', message: 'Transport not available' };
      }

      try {
        const originalImg = new Image();
        originalImg.crossOrigin = 'anonymous';
        await new Promise<void>((resolve, reject) => {
          originalImg.onload = () => resolve();
          originalImg.onerror = () => reject(new Error('Failed to load original image'));
          originalImg.src = source.originalImageSrc;
        });

        const originalWidth = originalImg.naturalWidth || originalImg.width || 512;
        const originalHeight = originalImg.naturalHeight || originalImg.height || 512;
        const factor = resolution === 'half' ? 2 : resolution === 'quarter' ? 4 : 1;
        const targetWidth = Math.floor(originalWidth / factor);
        const targetHeight = Math.floor(originalHeight / factor);

        const rasterized = await rasterizeImageToRgb(source.originalImageSrc, targetWidth, targetHeight);

        if (filters.length === 1 && filters[0].id === 'none') {
          onSourceUpdate(source.id, (current) => ({
            ...current,
            currentImageSrc: frameResponseToDataUrl(rasterized, targetWidth, targetHeight, 3),
            detections: [],
            detectionImageWidth: 0,
            detectionImageHeight: 0,
          }));
          return { status: 'success' };
        }

        const response = await source.transport.sendSingleImage(
          rasterized,
          targetWidth,
          targetHeight,
          3,
          mapFiltersToValueObjects(filters),
          accelerator,
          new GrayscaleAlgorithm('bt601')
        );

        logger.info('Static image response', {
          code: String(response.code),
          width: String(response.width),
          height: String(response.height),
          channels: String(response.channels),
          imageBytesLen: String(response.imageData?.byteLength ?? 0),
          detectionCount: String(response.detections?.length ?? 0),
          filters: filters.map((f) => f.id).join(','),
        });

        if (response.code === 0 && response.imageData?.byteLength && response.width > 0 && response.height > 0) {
          onSourceUpdate(source.id, (current) => ({
            ...current,
            currentImageSrc: frameResponseToDataUrl(
              response.imageData,
              response.width,
              response.height,
              response.channels || 4
            ),
            detections: response.detections,
            detectionImageWidth: response.width,
            detectionImageHeight: response.height,
          }));
          return { status: 'success' };
        }

        return {
          status: 'error',
          message: response.message || 'The image processor did not return a processed image.',
        };
      } catch (error) {
        logger.error('Error applying static image filter', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
        return {
          status: 'error',
          message: error instanceof Error ? error.message : 'Unknown processing error',
        };
      }
    },
    [mapFiltersToValueObjects, toastManager]
  );

  const applyVideoFilters = useCallback(
    async (options: FilterApplicationOptions): Promise<FilterApplicationResult> => {
      const { source, filters, accelerator } = options;

      if (!source.transport || !source.transport.isConnected()) {
        logger.error('Frame transport not connected for selected source', {
          'source.id': source.id,
        });
        return { status: 'error', message: 'Transport not connected' };
      }

      const videoId = source.videoId || source.id;
      pendingUpdateRef.current = { sourceId: source.id, filters };

      source.transport.sendStopVideo(videoId);

      await new Promise<void>((resolve) => {
        setTimeout(() => {
          const pending = pendingUpdateRef.current;
          if (!pending || pending.sourceId !== source.id) {
            resolve();
            return;
          }

          if (source.transport?.isConnected()) {
            source.transport.sendStartVideo(videoId, mapFiltersToValueObjects(pending.filters), accelerator);
            pendingUpdateRef.current = null;
          } else {
            toastManager.error(
              'Filter update failed',
              'Could not restart video stream. Please try selecting the video again.'
            );
            logger.error('Video filter update failed - transport not ready', {
              'source.id': source.id,
              'video.id': videoId,
            });
          }
          resolve();
        }, 200);
      });

      return { status: 'success' };
    },
    [mapFiltersToValueObjects, toastManager]
  );

  return { applyStaticFilters, applyVideoFilters };
}
