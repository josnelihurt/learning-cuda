import './register-mock-video-grid';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, act, waitFor } from '@testing-library/react';
import type { ReactElement } from 'react';
import { ToastProvider } from '../../context/toast-context';
import { GrpcClientsProvider } from '../../providers/grpc-clients-provider';
import { VideoGridHost } from './VideoGridHost';

const mockSetActiveFilters = vi.fn();
const mockSetResolution = vi.fn();
const mockSetSelectedSource = vi.fn();

const serviceMocks = vi.hoisted(() => ({
  listAvailableImages: vi.fn().mockResolvedValue([]),
  loggerError: vi.fn(),
}));

vi.mock('../../providers/app-services-provider', () => {
  const noopProcessorSvc = {
    isInitialized: () => true,
    addFiltersUpdatedListener: vi.fn(),
    removeFiltersUpdatedListener: vi.fn(),
  };
  return {
    useAppServices: () => ({
      container: {
        getInputSourceService: () => ({
          getSources: vi.fn().mockReturnValue([]),
          getDefaultSource: vi.fn().mockReturnValue(null),
          listAvailableImages: serviceMocks.listAvailableImages,
        }),
        getProcessorCapabilitiesService: () => noopProcessorSvc,
        getLogger: () => ({
          error: serviceMocks.loggerError,
          warn: vi.fn(),
          info: vi.fn(),
          debug: vi.fn(),
          shutdown: vi.fn(),
          initialize: vi.fn(),
        }),
      },
      ready: true,
    }),
  };
});

vi.mock('../../context/dashboard-state-context', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../context/dashboard-state-context')>();
  return {
    ...actual,
    useDashboardState: () => ({
      selectedSourceNumber: 1,
      selectedSourceName: 'A',
      selectedAccelerator: 'gpu',
      selectedResolution: 'original',
      activeFilters: [],
      processorFilterEpoch: 0,
      setSelectedSource: mockSetSelectedSource,
      setAccelerator: vi.fn(),
      setResolution: mockSetResolution,
      setActiveFilters: mockSetActiveFilters,
    }),
  };
});

import { DashboardStateProvider } from '../../context/dashboard-state-context';

function renderWithProviders(ui: ReactElement) {
  return render(
    <ToastProvider>
      <GrpcClientsProvider>
        <DashboardStateProvider>{ui}</DashboardStateProvider>
      </GrpcClientsProvider>
    </ToastProvider>
  );
}

function appendFabDrawerModal(): void {
  const fab = document.createElement('add-source-fab');
  const drawer = document.createElement('source-drawer') as HTMLElement & { open: () => void };
  drawer.open = vi.fn();
  const modal = document.createElement('image-selector-modal') as HTMLElement & {
    open: ReturnType<typeof vi.fn>;
  };
  modal.open = vi.fn();
  document.body.append(fab, drawer, modal);
}

describe('VideoGridHost', () => {
  beforeEach(() => {
    vi.spyOn(customElements, 'whenDefined').mockResolvedValue(undefined as never);
    mockSetActiveFilters.mockClear();
    mockSetResolution.mockClear();
    mockSetSelectedSource.mockClear();
    serviceMocks.listAvailableImages.mockClear();
    serviceMocks.listAvailableImages.mockResolvedValue([]);
    serviceMocks.loggerError.mockClear();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    document.body.innerHTML = '';
  });

  it('opens source drawer when add-source-fab dispatches open-drawer', async () => {
    const open = vi.fn();
    const fab = document.createElement('add-source-fab');
    const drawer = document.createElement('source-drawer') as HTMLElement & { open: typeof open };
    drawer.open = open;
    document.body.append(fab, drawer);

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    await act(async () => {
      fab.dispatchEvent(new CustomEvent('open-drawer', { bubbles: true, composed: true }));
    });

    expect(open).toHaveBeenCalled();
  });

  it('calls grid.addSource when source-drawer dispatches source-selected', async () => {
    const fab = document.createElement('add-source-fab');
    const drawer = document.createElement('source-drawer') as HTMLElement & { open: () => void };
    drawer.open = vi.fn();
    document.body.append(fab, drawer);

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    const gridEl = document.querySelector('video-grid') as HTMLElement & {
      addSource: ReturnType<typeof vi.fn>;
    };
    expect(gridEl).toBeTruthy();
    const addSpy = vi.spyOn(gridEl, 'addSource');

    const source = {
      id: 'cam-1',
      displayName: 'Camera',
      type: 'camera' as const,
      imagePath: '',
      isDefault: false,
    };

    await act(async () => {
      drawer.dispatchEvent(
        new CustomEvent('source-selected', {
          bubbles: true,
          composed: true,
          detail: { source },
        })
      );
    });

    expect(addSpy).toHaveBeenCalledWith(source);
  });

  it('syncs filters and resolution from source-selection-changed', async () => {
    appendFabDrawerModal();

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    const gridEl = document.querySelector('video-grid') as HTMLElement & {
      getSources: () => Array<{ id: string; name: string }>;
    };
    gridEl.getSources = () => [{ id: 's1', name: 'Test' }];

    await act(async () => {
      gridEl.dispatchEvent(
        new CustomEvent('source-selection-changed', {
          bubbles: true,
          composed: true,
          detail: {
            sourceId: 's1',
            sourceNumber: 2,
            filters: [{ id: 'blur', parameters: { radius: '3' } }],
            resolution: 'half',
          },
        })
      );
    });

    expect(mockSetSelectedSource).toHaveBeenCalledWith(2, 'Test');
    expect(mockSetActiveFilters).toHaveBeenCalledWith([
      { id: 'blur', parameters: { radius: '3' } },
    ]);
    expect(mockSetResolution).toHaveBeenCalledWith('half');
  });

  it('opens image selector on change-image-requested and applies selection', async () => {
    const sampleImages = [{ id: 'i1', path: '/x.png', displayName: 'X', isDefault: false }];
    serviceMocks.listAvailableImages.mockResolvedValue(sampleImages);
    appendFabDrawerModal();

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    const gridEl = document.querySelector('video-grid') as HTMLElement & {
      changeSourceImage: ReturnType<typeof vi.fn>;
    };
    const modal = document.querySelector('image-selector-modal') as HTMLElement & {
      open: ReturnType<typeof vi.fn>;
    };
    const changeSpy = vi.spyOn(gridEl, 'changeSourceImage');

    await act(async () => {
      gridEl.dispatchEvent(
        new CustomEvent('change-image-requested', {
          bubbles: true,
          composed: true,
          detail: { sourceId: 's-1', sourceNumber: 2 },
        })
      );
    });

    await waitFor(() => {
      expect(serviceMocks.listAvailableImages).toHaveBeenCalled();
      expect(modal.open).toHaveBeenCalledWith(sampleImages);
    });

    await act(async () => {
      modal.dispatchEvent(
        new CustomEvent('image-selected', {
          bubbles: true,
          composed: true,
          detail: { image: { path: '/chosen.png' } },
        })
      );
    });

    expect(changeSpy).toHaveBeenCalledWith(2, '/chosen.png');
  });

  it('logs when listAvailableImages fails for change-image', async () => {
    serviceMocks.listAvailableImages.mockRejectedValue(new Error('network'));
    appendFabDrawerModal();

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    const gridEl = document.querySelector('video-grid') as HTMLElement;

    await act(async () => {
      gridEl.dispatchEvent(
        new CustomEvent('change-image-requested', {
          bubbles: true,
          composed: true,
          detail: { sourceId: 's-1', sourceNumber: 1 },
        })
      );
    });

    await waitFor(() => {
      expect(serviceMocks.loggerError).toHaveBeenCalled();
    });
  });
});
