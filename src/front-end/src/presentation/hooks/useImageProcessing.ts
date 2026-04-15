import { useState, useCallback, useRef, useEffect } from 'react';
import { useServiceContext } from '../context/service-context';
import { useToast, type ToastApi } from './useToast';
import type { ActiveFilterState } from '../components/filters/FilterPanel';
import {
  ProcessImageRequest,
  ProcessImageResponse,
  GenericFilterSelection,
  GenericFilterParameterSelection,
} from '@/gen/image_processor_service_pb';
import { logger } from '@/infrastructure/observability/otel-logger';

export interface UseImageProcessingReturn {
  processing: boolean;
  progress: number;
  processedImageUrl: string | null;
  error: string | null;
  processImage: (imagePath: string, activeFilters: ActiveFilterState[]) => Promise<void>;
  reset: () => void;
}

export function useImageProcessing(): UseImageProcessingReturn {
  const { imageProcessorClient } = useServiceContext();
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [processedImageUrl, setProcessedImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const toastApi = useToast() as ToastApi;

  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const requestGenRef = useRef(0);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  const reset = useCallback(() => {
    setProcessedImageUrl(null);
    setError(null);
    setProgress(0);
  }, []);

  const processImage = useCallback(
    async (imagePath: string, activeFilters: ActiveFilterState[]) => {
      const gen = ++requestGenRef.current;

      // Clear any existing progress interval
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
        progressIntervalRef.current = null;
      }

      // Reset state
      setError(null);
      setProcessedImageUrl(null);
      setProgress(0);

      // Validation: check if image path is provided
      if (!imagePath || imagePath.trim() === '') {
        const errorMsg = 'No image selected. Please upload or select an image first.';
        setError(errorMsg);
        toastApi.error('No Image Selected', errorMsg);
        return;
      }

      // Validation: check if filters are selected
      if (!activeFilters || activeFilters.length === 0) {
        const errorMsg = 'No filters selected. Please select at least one filter.';
        setError(errorMsg);
        toastApi.error('No Filters Selected', errorMsg);
        return;
      }

      // Check if filter selection is "none" placeholder
      const hasValidFilters = activeFilters.some(
        (filter) => filter.id !== 'none' && Object.keys(filter.parameters).length > 0
      );

      if (!hasValidFilters) {
        const errorMsg = 'No filters selected. Please enable and configure at least one filter.';
        setError(errorMsg);
        toastApi.error('No Filters Selected', errorMsg);
        return;
      }

      setProcessing(true);
      setProgress(0);

      // Start progress simulation (0-90% during fetch/process)
      progressIntervalRef.current = setInterval(() => {
        setProgress((prev) => {
          if (prev < 90) {
            return prev + 5;
          }
          return prev;
        });
      }, 100);

      try {
        logger.debug('Starting image processing', { imagePath, filterCount: activeFilters.length });

        // Fetch image data from the path
        const response = await fetch(imagePath);
        if (!response.ok) {
          throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }

        const imageBlob = await response.blob();
        const imageData = await imageBlob.arrayBuffer();

        // Create an Image object to get dimensions
        const img = new Image();
        const imageUrl = URL.createObjectURL(imageBlob);

        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error('Failed to load image'));
          img.src = imageUrl;
        });

        const width = img.width;
        const height = img.height;
        const channels = 4; // RGBA

        // Clean up object URL
        URL.revokeObjectURL(imageUrl);

        // Convert ActiveFilterState[] to GenericFilterSelection[]
        const genericFilters: GenericFilterSelection[] = activeFilters
          .filter((filter) => filter.id !== 'none')
          .map((filter) => {
            const parameterSelections: GenericFilterParameterSelection[] = Object.entries(
              filter.parameters
            ).map(([parameterId, value]) => {
              const selection = new GenericFilterParameterSelection();
              selection.parameterId = parameterId;
              selection.values = [value];
              return selection;
            });

            const selection = new GenericFilterSelection();
            selection.filterId = filter.id;
            selection.parameters = parameterSelections;
            return selection;
          });

        // Create the process image request
        const request = new ProcessImageRequest();
        request.imageData = new Uint8Array(imageData);
        request.width = width;
        request.height = height;
        request.channels = channels;
        request.genericFilters = genericFilters;
        request.apiVersion = '1.0';

        // Call the gRPC service
        const responseMsg: ProcessImageResponse = await imageProcessorClient.processImage(request);

        // Check if request was aborted or superseded
        if (gen !== requestGenRef.current) {
          return;
        }

        // Clear progress interval
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
          progressIntervalRef.current = null;
        }

        // Check response
        if (responseMsg.code !== 0) {
          throw new Error(responseMsg.message || 'Image processing failed');
        }

        // Create a blob URL for the processed image
        const processedBlob = new Blob([responseMsg.imageData], { type: 'image/png' });
        const processedUrl = URL.createObjectURL(processedBlob);

        setProcessedImageUrl(processedUrl);
        setProgress(100);

        const successMessage = responseMsg.message || 'Image processed successfully';
        toastApi.success('Processing Complete', successMessage);

        logger.info('Image processed successfully', {
          width: responseMsg.width,
          height: responseMsg.height,
          channels: responseMsg.channels,
        });

        // Reset processing state after short delay
        setTimeout(() => {
          if (gen === requestGenRef.current) {
            setProcessing(false);
            setProgress(0);
          }
        }, 300);
      } catch (err) {
        // Clear progress interval
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current);
          progressIntervalRef.current = null;
        }

        // Check if request was aborted or superseded
        if (gen !== requestGenRef.current) {
          return;
        }

        const errorMsg = err instanceof Error ? err.message : 'Image processing failed';
        setError(errorMsg);
        setProcessing(false);
        setProgress(0);

        toastApi.error('Processing Failed', errorMsg);

        logger.error('Image processing failed', { error: err });
      }
    },
    [imageProcessorClient, toastApi]
  );

  return {
    processing,
    progress,
    processedImageUrl,
    error,
    processImage,
    reset,
  };
}
