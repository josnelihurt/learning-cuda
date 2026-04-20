import { useCallback, useState } from 'react';
import { fileService } from '@/infrastructure/data/file-service';
import { useToast, type ToastApi } from './useToast';
import type { StaticImage } from '@/gen/common_pb';

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

interface UseImageUploadReturn {
  uploading: boolean;
  progress: number;
  error: string | null;
  uploadFile: (file: File) => Promise<StaticImage | undefined>;
}

export function useImageUpload(): UseImageUploadReturn {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const toastApi = useToast() as ToastApi;

  const uploadFile = useCallback(
    async (file: File): Promise<StaticImage | undefined> => {
      // Reset state
      setError(null);
      setProgress(0);

      // Validate file type
      if (!file.name.toLowerCase().endsWith('.png')) {
        const errorMsg = 'Only PNG files are supported';
        setError(errorMsg);
        toastApi.error('Invalid File', errorMsg);
        return undefined;
      }

      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        const errorMsg = 'File size exceeds 10MB limit';
        setError(errorMsg);
        toastApi.error('File Too Large', errorMsg);
        return undefined;
      }

      setUploading(true);
      setProgress(0);

      // Simulate progress (will be updated during actual upload)
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev < 90) {
            return prev + 10;
          }
          return prev;
        });
      }, 100);

      try {
        const image = await fileService.uploadImage(file);
        clearInterval(progressInterval);
        setProgress(100);

        // Reset state after short delay
        setTimeout(() => {
          setUploading(false);
          setProgress(0);
        }, 300);

        return image;
      } catch (err) {
        clearInterval(progressInterval);
        setUploading(false);
        setProgress(0);

        const errorMsg = err instanceof Error ? err.message : 'Upload failed';
        setError(errorMsg);
        toastApi.error('Upload Failed', errorMsg);

        return undefined;
      }
    },
    [toastApi]
  );

  return {
    uploading,
    progress,
    error,
    uploadFile,
  };
}
