import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { InputSource } from '@/gen/config_service_pb';
import type { VideoGrid } from '@/components/video/video-grid';
import type { StatsPanel } from '@/components/app/stats-panel';
import type { SourceDrawer } from '@/components/app/source-drawer';
import type { ImageSelectorModal } from '@/components/image/image-selector-modal';
import type { ToastContainer } from '@/components/app/toast-container';
import type { ActiveFilterState } from '../filters/FilterPanel';
import { useAppServices } from '../../providers/app-services-provider';
import { useDashboardState } from '../../context/dashboard-state-context';

type SourceSelectionDetail = {
  sourceId: string;
  sourceNumber: number;
  sourceType?: string;
  filters?: ActiveFilterState[];
  resolution?: string;
};

export function VideoGridHost() {
  const [grid, setGrid] = useState<VideoGrid | null>(null);
  const pendingSourceNumberForImageChangeRef = useRef<number | null>(null);
  const { container, ready } = useAppServices();
  const {
    activeFilters,
    selectedAccelerator,
    selectedResolution,
    setSelectedSource,
    setActiveFilters,
    setResolution,
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
      const ce = e as CustomEvent<SourceSelectionDetail>;
      const id = ce.detail?.sourceId;
      const num = ce.detail?.sourceNumber;
      const src = grid.getSources().find((s) => s.id === id);
      setSelectedSource(num, src?.name ?? '');
      if (ce.detail?.filters !== undefined) {
        setActiveFilters(
          ce.detail.filters.map((f) => ({
            id: f.id,
            parameters: { ...f.parameters },
          }))
        );
      }
      if (ce.detail?.resolution !== undefined) {
        setResolution(ce.detail.resolution);
      }
    }) as EventListener;
    grid.addEventListener('source-selection-changed', onSelection);
    return () => grid.removeEventListener('source-selection-changed', onSelection);
  }, [grid, ready, setSelectedSource, setActiveFilters, setResolution]);

  useEffect(() => {
    if (!grid || !ready) {
      return;
    }
    const fab = document.querySelector('add-source-fab');
    const drawer = document.querySelector('source-drawer') as SourceDrawer | null;
    const inputSourceService = container.getInputSourceService();

    const onOpenDrawer = (): void => {
      if (drawer && inputSourceService) {
        const sources = inputSourceService.getSources();
        const selectedIds = grid.getSelectedSourceIds();
        drawer.open(sources, selectedIds);
      }
    };

    const onSourceSelected = (e: Event): void => {
      const ce = e as CustomEvent<{ source: InputSource }>;
      if (ce.detail?.source) {
        grid.addSource(ce.detail.source);
      }
    };

    fab?.addEventListener('open-drawer', onOpenDrawer);
    drawer?.addEventListener('source-selected', onSourceSelected as EventListener);

    return () => {
      fab?.removeEventListener('open-drawer', onOpenDrawer);
      drawer?.removeEventListener('source-selected', onSourceSelected as EventListener);
    };
  }, [grid, ready, container]);

  useEffect(() => {
    if (!grid || !ready) {
      return;
    }
    const modal = document.querySelector('image-selector-modal') as ImageSelectorModal | null;
    const inputSourceService = container.getInputSourceService();
    const logger = container.getLogger();

    const onChangeImageRequested = (e: Event): void => {
      const ce = e as CustomEvent<{ sourceId?: string; sourceNumber?: number }>;
      pendingSourceNumberForImageChangeRef.current = ce.detail?.sourceNumber ?? null;
      if (!modal || !inputSourceService || pendingSourceNumberForImageChangeRef.current === null) {
        return;
      }
      void (async () => {
        try {
          const images = await inputSourceService.listAvailableImages();
          modal.open(images);
        } catch (error) {
          logger.error('Failed to load images for image selector', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          pendingSourceNumberForImageChangeRef.current = null;
        }
      })();
    };

    const onImageSelected = (e: Event): void => {
      const ce = e as CustomEvent<{ image: { path: string } }>;
      const imagePath = ce.detail?.image?.path;
      const n = pendingSourceNumberForImageChangeRef.current;
      if (imagePath !== undefined && n !== null) {
        grid.changeSourceImage(n, imagePath);
        pendingSourceNumberForImageChangeRef.current = null;
      }
    };

    grid.addEventListener('change-image-requested', onChangeImageRequested);
    modal?.addEventListener('image-selected', onImageSelected as EventListener);

    return () => {
      grid.removeEventListener('change-image-requested', onChangeImageRequested);
      modal?.removeEventListener('image-selected', onImageSelected as EventListener);
    };
  }, [grid, ready, container]);

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
