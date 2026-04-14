import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, act, waitFor, screen, fireEvent } from '@testing-library/react';
import type { ReactElement } from 'react';
import { ToastProvider } from '../../context/toast-context';
import { GrpcClientsProvider } from '../../providers/grpc-clients-provider';
import { VideoGridHost } from './VideoGridHost';

const mockSetActiveFilters = vi.fn();
const mockSetResolution = vi.fn();
const mockSetSelectedSource = vi.fn();

const serviceMocks = vi.hoisted(() => ({
  getSources: vi.fn().mockReturnValue([]),
  getDefaultSource: vi.fn().mockReturnValue(null),
  listAvailableImages: vi.fn().mockResolvedValue([]),
  loggerError: vi.fn(),
}));

vi.mock('@/infrastructure/transport/websocket-frame-transport', () => ({
  WebSocketService: class {
    connect() {}
    disconnect() {}
    isConnected() {
      return true;
    }
    setProcessing() {}
    onFrameResult() {}
    sendStartVideo() {}
    sendStopVideo() {}
    sendFrame() {}
    async sendSingleFrame() {
      return { success: true, response: { imageData: new Uint8Array() } };
    }
  },
}));

vi.mock('@/infrastructure/connection/webrtc-service', () => ({
  webrtcService: {
    createSession: vi.fn().mockResolvedValue({ getId: () => 'session-1' }),
    stopHeartbeat: vi.fn(),
    closeSession: vi.fn().mockResolvedValue(undefined),
  },
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
          getSources: serviceMocks.getSources,
          getDefaultSource: serviceMocks.getDefaultSource,
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

describe('VideoGridHost', () => {
  beforeEach(() => {
    vi.spyOn(customElements, 'whenDefined').mockResolvedValue(undefined as never);
    mockSetActiveFilters.mockClear();
    mockSetResolution.mockClear();
    mockSetSelectedSource.mockClear();
    serviceMocks.getSources.mockClear();
    serviceMocks.getDefaultSource.mockClear();
    serviceMocks.getSources.mockReturnValue([]);
    serviceMocks.getDefaultSource.mockReturnValue(null);
    serviceMocks.listAvailableImages.mockClear();
    serviceMocks.listAvailableImages.mockResolvedValue([]);
    serviceMocks.loggerError.mockClear();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    document.body.innerHTML = '';
  });

  it('opens source drawer when add-input button is clicked', async () => {
    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('add-input-fab'));
    });

    const drawer = screen.getByTestId('source-drawer');
    expect(drawer.className).toContain('show');
  });

  it('adds a source when selecting one from drawer', async () => {
    serviceMocks.getSources.mockReturnValue([
      {
        id: 'img-1',
        displayName: 'Image',
        type: 'static',
        imagePath: '/image.png',
        isDefault: false,
      },
    ]);

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('add-input-fab'));
    });
    fireEvent.click(screen.getByTestId('source-item-img-1'));

    expect(screen.getByTestId('source-card-1')).toBeTruthy();
  });

  it('syncs selected source to dashboard state when adding first source', async () => {
    serviceMocks.getSources.mockReturnValue([
      {
        id: 's1',
        displayName: 'Test',
        type: 'static',
        imagePath: '/test.png',
        isDefault: false,
      },
    ]);

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('add-input-fab'));
    });
    fireEvent.click(screen.getByTestId('source-item-s1'));

    expect(mockSetSelectedSource).toHaveBeenCalledWith(1, 'Test');
    expect(mockSetActiveFilters).toHaveBeenCalledWith([
      { id: 'none', parameters: {} },
    ]);
    expect(mockSetResolution).toHaveBeenCalledWith('original');
  });

  it('opens image selector and applies selected image for a static source', async () => {
    const sampleImages = [{ id: 'i1', path: '/x.png', displayName: 'X', isDefault: false }];
    serviceMocks.listAvailableImages.mockResolvedValue(sampleImages);
    serviceMocks.getSources.mockReturnValue([
      {
        id: 's-1',
        displayName: 'Static',
        type: 'static',
        imagePath: '/before.png',
        isDefault: false,
      },
    ]);

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('add-input-fab'));
    });
    fireEvent.click(screen.getByTestId('source-item-s-1'));
    fireEvent.click(screen.getByTestId('change-image-button'));

    await waitFor(() => {
      expect(serviceMocks.listAvailableImages).toHaveBeenCalled();
      expect(screen.getByTestId('image-selector-modal').className).toContain('show');
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('image-item-i1'));
    });

    const image = screen.getByAltText('Static') as HTMLImageElement;
    expect(image.src).toContain('/x.png');
  });

  it('logs when listAvailableImages fails for image selector request', async () => {
    serviceMocks.listAvailableImages.mockRejectedValue(new Error('network'));
    serviceMocks.getSources.mockReturnValue([
      {
        id: 's-1',
        displayName: 'Static',
        type: 'static',
        imagePath: '/before.png',
        isDefault: false,
      },
    ]);

    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    await act(async () => {
      fireEvent.click(screen.getByTestId('add-input-fab'));
    });
    fireEvent.click(screen.getByTestId('source-item-s-1'));
    fireEvent.click(screen.getByTestId('change-image-button'));

    await waitFor(() => {
      expect(serviceMocks.loggerError).toHaveBeenCalledWith(
        'Failed to load image options',
        expect.any(Object)
      );
    });
  });
});
