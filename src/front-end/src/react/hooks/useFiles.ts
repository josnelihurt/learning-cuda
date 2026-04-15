import { useCallback, useEffect, useRef, useState } from 'react';
import { ConnectError } from '@connectrpc/connect';
import type { StaticImage } from '@/gen/common_pb';
import { fileService } from '@/infrastructure/data/file-service';
import { useToast } from './useToast';

export function useFiles() {
  const [images, setImages] = useState<StaticImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestGenRef = useRef(0);
  const toast = useToast();

  const fetchImages = useCallback(() => {
    abortRef.current?.abort();
    const gen = ++requestGenRef.current;
    const ac = new AbortController();
    abortRef.current = ac;
    const { signal } = ac;

    setLoading(true);
    setError(null);

    void (async () => {
      try {
        const response = await fileService.listAvailableImages();
        if (signal.aborted || gen !== requestGenRef.current) {
          return;
        }
        setImages(response);
      } catch (e) {
        if (signal.aborted || gen !== requestGenRef.current) {
          return;
        }
        const conn = ConnectError.from(e);
        const errorMessage = conn.message || 'Failed to load images';
        setError(errorMessage);
        toast.error('Error', errorMessage);
      } finally {
        if (gen === requestGenRef.current) {
          setLoading(false);
        }
      }
    })();
  }, [toast]);

  useEffect(() => {
    fetchImages();
    return () => {
      abortRef.current?.abort();
    };
  }, [fetchImages]);

  return { images, loading, error, fetchImages };
}
