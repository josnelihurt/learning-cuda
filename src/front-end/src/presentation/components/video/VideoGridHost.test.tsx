import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, act, screen, fireEvent } from '@testing-library/react';
import type { ReactElement } from 'react';
import { ToastProvider } from '@/presentation/context/toast-context';
import { GrpcClientsProvider } from '@/presentation/providers/grpc-clients-provider';
import { DashboardStateProvider } from '@/presentation/context/dashboard-state-context';
import { VideoGridHost } from './VideoGridHost';

const serviceMocks = vi.hoisted(() => ({
  getSources: vi.fn().mockReturnValue([]),
  getDefaultSource: vi.fn().mockReturnValue(null),
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
    serviceMocks.getSources.mockClear();
    serviceMocks.getDefaultSource.mockClear();
    serviceMocks.listAvailableImages.mockClear();
    serviceMocks.loggerError.mockClear();
    serviceMocks.getSources.mockReturnValue([]);
    serviceMocks.getDefaultSource.mockReturnValue(null);
    serviceMocks.listAvailableImages.mockResolvedValue([]);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    document.body.innerHTML = '';
  });

  it('renders host shell and add button', async () => {
    await act(async () => {
      renderWithProviders(<VideoGridHost />);
    });

    expect(screen.getByTestId('video-grid')).toBeTruthy();
    expect(screen.getByTestId('add-input-fab')).toBeTruthy();
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
    expect(serviceMocks.getSources).toHaveBeenCalled();
  });
});
