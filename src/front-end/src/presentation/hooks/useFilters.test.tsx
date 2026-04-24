import { describe, it, expect, vi, afterEach } from 'vitest';
import { useEffect } from 'react';
import { act } from 'react';
import {
  GenericFilterDefinition,
  ListFiltersResponse,
} from '@/gen/image_processor_service_pb';
import { useFilters } from '@/presentation/hooks/useFilters';
import { renderWithService } from '@/presentation/test-utils/render-with-service';
import type { GrpcClients } from '@/presentation/context/service-context';

afterEach(() => {
  document.body.replaceChildren();
});

describe('useFilters', () => {
  it('loads filters via listFilters, then refetch increases call count', async () => {
    const listFilters = vi.fn().mockResolvedValue(
      new ListFiltersResponse({
        filters: [new GenericFilterDefinition({ id: 'blur', name: 'Blur' })],
      })
    );

    const snapshots: {
      loading: boolean;
      filtersLen: number;
      firstId?: string;
    }[] = [];
    let refetchFn: (() => void) | undefined;

    function Probe(): React.ReactNode {
      const { filters, loading, refetch } = useFilters();
      refetchFn = refetch;
      useEffect(() => {
        snapshots.push({
          loading,
          filtersLen: filters.length,
          firstId: filters[0]?.id,
        });
      }, [loading, filters]);
      return null;
    }

    const { unmount } = renderWithService(<Probe />, {
      imageProcessorClient: { listFilters } as unknown as GrpcClients['imageProcessorClient'],
      remoteManagementClient: {} as GrpcClients['remoteManagementClient'],
    });

    await act(async () => {
      await vi.waitFor(() => {
        expect(listFilters).toHaveBeenCalled();
      });
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(listFilters).toHaveBeenCalledTimes(1);
    const done = snapshots.filter((s) => !s.loading).pop();
    expect(done?.filtersLen).toBe(1);
    expect(done?.firstId).toBe('blur');

    await act(async () => {
      refetchFn?.();
      await Promise.resolve();
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(listFilters).toHaveBeenCalledTimes(2);
    unmount();
  });
});
