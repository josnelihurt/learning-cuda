import { useCallback, useEffect, useRef, useState } from 'react';
import { ConnectError } from '@connectrpc/connect';
import {
  AcceleratorHealthStatus,
  CheckAcceleratorHealthRequest,
  type CheckAcceleratorHealthResponse,
} from '@/gen/remote_management_service_pb';
import { useServiceContext } from '../context/service-context';
import type { GrpcAsyncError } from './useAsyncGRPC';

export function useHealthMonitor(options?: { pollIntervalMs?: number }) {
  // D-14: default poll in [15000, 30000] ms; 20000 balances traffic vs freshness.
  const pollIntervalMs = options?.pollIntervalMs ?? 20000;

  const { remoteManagementClient } = useServiceContext();
  const clientRef = useRef(remoteManagementClient);
  clientRef.current = remoteManagementClient;

  const [response, setResponse] = useState<CheckAcceleratorHealthResponse | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<GrpcAsyncError | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | undefined>();

  const abortRef = useRef<AbortController | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const requestGenRef = useRef(0);
  const mountedRef = useRef(true);

  const isHealthy = response?.status === AcceleratorHealthStatus.HEALTHY;

  const check = useCallback(() => {
    abortRef.current?.abort();
    const gen = ++requestGenRef.current;
    const ac = new AbortController();
    abortRef.current = ac;
    const { signal } = ac;

    setLoading(true);
    setError(null);

    void (async () => {
      try {
        const res = await clientRef.current.checkAcceleratorHealth(
          new CheckAcceleratorHealthRequest({}),
          { signal }
        );
        if (
          signal.aborted ||
          gen !== requestGenRef.current ||
          !mountedRef.current
        ) {
          return;
        }
        setResponse(res);
        setLastChecked(new Date());
      } catch (e) {
        if (
          signal.aborted ||
          gen !== requestGenRef.current ||
          !mountedRef.current
        ) {
          return;
        }
        const conn = ConnectError.from(e);
        setError({ message: conn.message, code: String(conn.code) });
      } finally {
        if (gen === requestGenRef.current && mountedRef.current) {
          setLoading(false);
        }
      }
    })();
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    const clearPolling = () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };

    const startPolling = () => {
      clearPolling();
      if (document.visibilityState !== 'visible') {
        return;
      }
      intervalRef.current = setInterval(() => {
        check();
      }, pollIntervalMs);
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        clearPolling();
        abortRef.current?.abort();
        return;
      }
      check();
      startPolling();
    };

    if (document.visibilityState === 'visible') {
      check();
      startPolling();
    }

    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      mountedRef.current = false;
      document.removeEventListener('visibilitychange', onVisibilityChange);
      clearPolling();
      abortRef.current?.abort();
    };
  }, [check, pollIntervalMs]);

  return {
    response,
    isHealthy,
    loading,
    error,
    lastChecked,
  };
}
