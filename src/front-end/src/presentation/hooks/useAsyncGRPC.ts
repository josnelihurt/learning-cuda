import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DependencyList,
} from 'react';
import { ConnectError } from '@connectrpc/connect';
import { useServiceContext } from '../context/service-context';
import type { GrpcClients } from '../context/service-context';

export type GrpcAsyncError = { message: string; code?: string };

export function useAsyncGRPC<T>(
  executor: (clients: GrpcClients, opts: { signal: AbortSignal }) => Promise<T>,
  deps: DependencyList = []
) {
  const clients = useServiceContext();
  const clientsRef = useRef(clients);
  clientsRef.current = clients;
  const executorRef = useRef(executor);
  executorRef.current = executor;

  const [data, setData] = useState<T | undefined>(undefined);
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
        const result = await executorRef.current(clientsRef.current, { signal });
        if (signal.aborted || gen !== requestGenRef.current) {
          return;
        }
        setData(result);
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
  }, [refetch, ...deps]);

  return { data, loading, error, refetch };
}
