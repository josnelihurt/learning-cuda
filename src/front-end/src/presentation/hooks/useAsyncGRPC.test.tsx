import { describe, it, expect, vi, afterEach } from 'vitest';
import { useEffect } from 'react';
import { act } from 'react';
import { ConnectError, Code } from '@connectrpc/connect';
import { CheckAcceleratorHealthResponse, AcceleratorHealthStatus } from '@/gen/remote_management_service_pb';
import { useAsyncGRPC } from '@/presentation/hooks/useAsyncGRPC';
import { renderWithService } from '@/presentation/test-utils/render-with-service';
import type { GrpcClients } from '@/presentation/context/service-context';

afterEach(() => {
  document.body.replaceChildren();
});

describe('useAsyncGRPC', () => {
  it('transitions loading to data on success', async () => {
    const checkHealth = vi.fn().mockResolvedValue(
      new CheckAcceleratorHealthResponse({ status: AcceleratorHealthStatus.HEALTHY })
    );
    const snapshots: { loading: boolean; data?: unknown }[] = [];

    function Probe(): React.ReactNode {
      const { data, loading } = useAsyncGRPC(
        async (clients, { signal }) =>
          clients.remoteManagementClient.checkAcceleratorHealth({}, { signal }),
        []
      );
      snapshots.push({ loading, data });
      return null;
    }

    const { unmount } = renderWithService(<Probe />, {
      remoteManagementClient: { checkAcceleratorHealth: checkHealth } as unknown as GrpcClients['remoteManagementClient'],
    });

    await act(async () => {
      await vi.waitFor(() => {
        expect(checkHealth).toHaveBeenCalled();
      });
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(checkHealth).toHaveBeenCalled();
    const last = snapshots[snapshots.length - 1];
    expect(last?.loading).toBe(false);
    expect(last?.data).toBeDefined();
    unmount();
  });

  it('maps ConnectError to stable error shape', async () => {
    const checkHealth = vi
      .fn()
      .mockRejectedValue(new ConnectError('rpc failed', Code.Unavailable));
    const errors: { message: string; code?: string }[] = [];

    function Probe(): React.ReactNode {
      const { error, loading } = useAsyncGRPC(
        async (clients, { signal }) =>
          clients.remoteManagementClient.checkAcceleratorHealth({}, { signal }),
        []
      );
      useEffect(() => {
        if (!loading && error) {
          errors.push(error);
        }
      }, [loading, error]);
      return null;
    }

    const { unmount } = renderWithService(<Probe />, {
      remoteManagementClient: { checkAcceleratorHealth: checkHealth } as unknown as GrpcClients['remoteManagementClient'],
    });

    await vi.waitFor(
      () => {
        expect(errors.length).toBeGreaterThan(0);
      },
      { timeout: 3000 }
    );

    expect(errors[0]?.message).toBeDefined();
    expect(errors[0]?.code).toBe(String(Code.Unavailable));
    unmount();
  });

  it('refetch runs executor again', async () => {
    const checkHealth = vi.fn().mockResolvedValue(
      new CheckAcceleratorHealthResponse({ status: AcceleratorHealthStatus.HEALTHY })
    );
    let refetch: (() => void) | undefined;

    function Probe(): React.ReactNode {
      const grpc = useAsyncGRPC(
        async (clients, { signal }) =>
          clients.remoteManagementClient.checkAcceleratorHealth({}, { signal }),
        []
      );
      refetch = grpc.refetch;
      return null;
    }

    const { unmount } = renderWithService(<Probe />, {
      remoteManagementClient: { checkAcceleratorHealth: checkHealth } as unknown as GrpcClients['remoteManagementClient'],
    });

    await act(async () => {
      await vi.waitFor(() => expect(checkHealth).toHaveBeenCalledTimes(1));
    });

    await act(async () => {
      refetch?.();
    });

    await act(async () => {
      await vi.waitFor(() => expect(checkHealth).toHaveBeenCalledTimes(2));
    });

    unmount();
  });
});
