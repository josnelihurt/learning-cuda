import { useCallback, useState } from 'react';
import { fileService } from '@/infrastructure/data/file-service';
import { useToast } from './useToast';
import type { StaticImage } from '@/gen/config_service_pb';

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export interface UseImageUploadReturn {
  uploading: boolean;
  progress: number;
  error: string | null;
  uploadFile: (file: File) => Promise<StaticImage | undefined>;
}

export function useImageUpload(): UseImageUploadReturn {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const { error: showError } = useToast();

  const uploadFile = useCallback(
    async (file: File): Promise<StaticImage | undefined> => {
      // Reset state
      setError(null);
      setProgress(0);

      // Validate file type
      if (!file.name.toLowerCase().endsWith('.png')) {
        const errorMsg = 'Only PNG files are supported';
        setError(errorMsg);
        showError('Invalid File', errorMsg);
        return undefined;
      }

      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        const errorMsg = 'File size exceeds 10MB limit';
        setError(errorMsg);
        showError('File Too Large', errorMsg);
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
        showError('Upload Failed', errorMsg);

        return undefined;
      }
    },
    [showError]
  );

  return {
    uploading,
    progress,
    error,
    uploadFile,
  };
}
