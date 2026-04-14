import { useCallback, useEffect, useRef, useState } from 'react';
import { ConnectError } from '@connectrpc/connect';
import {
  ListFiltersRequest,
  type GenericFilterDefinition,
} from '@/gen/image_processor_service_pb';
import { useServiceContext } from '../context/service-context';
import type { GrpcAsyncError } from './useAsyncGRPC';

export function useFilters() {
  const { imageProcessorClient } = useServiceContext();
  const clientRef = useRef(imageProcessorClient);
  clientRef.current = imageProcessorClient;

  const [filters, setFilters] = useState<GenericFilterDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<GrpcAsyncError | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestGenRef = useRef(0);

  const refetch = useCallback(() => {
    abortRef.current?.abort();
    const gen = ++requestGenRef.current;
    const ac = new AbortController();
    abortRef.current = ac;
    const { signal } = ac;

    setLoading(true);
    setError(null);

    void (async () => {
      try {
        const response = await clientRef.current.listFilters(
          new ListFiltersRequest({}),
          { signal }
        );
        if (signal.aborted || gen !== requestGenRef.current) {
          return;
        }
        setFilters(response.filters);
      } catch (e) {
        if (signal.aborted || gen !== requestGenRef.current) {
          return;
        }
        const conn = ConnectError.from(e);
        setError({ message: conn.message, code: String(conn.code) });
      } finally {
        if (gen === requestGenRef.current) {
          setLoading(false);
        }
      }
    })();
  }, []);

  useEffect(() => {
    refetch();
    return () => {
      abortRef.current?.abort();
    };
  }, [refetch]);

  return { filters, loading, error, refetch };
}
