import { useEffect, useLayoutEffect, useState } from 'react';
import type { VideoGrid } from '@/components/video/video-grid';
import type { StatsPanel } from '@/components/app/stats-panel';
import type { ToastContainer } from '@/components/app/toast-container';
import { useAppServices } from '../../providers/app-services-provider';
import { useDashboardState } from '../../context/dashboard-state-context';

export function VideoGridHost() {
  const [grid, setGrid] = useState<VideoGrid | null>(null);
  const { container, ready } = useAppServices();
  const {
    activeFilters,
    selectedAccelerator,
    selectedResolution,
    setSelectedSource,
  } = useDashboardState();

  useLayoutEffect(() => {
    if (!ready || !grid) {
      return;
    }
    const stats = document.querySelector('stats-panel') as StatsPanel | null;
    const toast = document.querySelector('toast-container') as ToastContainer | null;
    if (stats && toast) {
      grid.setManagers(stats, toast);
    }
  }, [grid, ready]);

  useEffect(() => {
    if (!ready || !grid) {
      return;
    }
    let cancelled = false;
    void (async () => {
      await customElements.whenDefined('video-grid');
      if (cancelled) {
        return;
      }
      const input = container.getInputSourceService();
      const defaultSource = input.getDefaultSource();
      if (defaultSource && grid.getSources().length === 0) {
        grid.addSource(defaultSource);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [container, grid, ready]);

  useEffect(() => {
    if (!grid || !ready) {
      return;
    }
    const onSelection = ((e: Event) => {
      const ce = e as CustomEvent<{ sourceId: string; sourceNumber: number }>;
      const id = ce.detail?.sourceId;
      const num = ce.detail?.sourceNumber;
      const src = grid.getSources().find((s) => s.id === id);
      setSelectedSource(num, src?.name ?? '');
    }) as EventListener;
    grid.addEventListener('source-selection-changed', onSelection);
    return () => grid.removeEventListener('source-selection-changed', onSelection);
  }, [grid, ready, setSelectedSource]);

  useEffect(() => {
    if (!grid || !ready) {
      return;
    }
    void grid.applyFilterToSelected(activeFilters, selectedAccelerator, selectedResolution);
  }, [grid, activeFilters, selectedAccelerator, selectedResolution, ready]);

  return (
    <video-grid
      ref={(el: HTMLElement | null) => {
        setGrid(el as VideoGrid | null);
      }}
      data-testid="video-grid-host"
    />
  );
}
