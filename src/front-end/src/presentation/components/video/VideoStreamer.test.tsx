import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import { VideoStreamer } from './VideoStreamer';

const dashboardFiltersRef: { current: ActiveFilterState[] } = { current: [] };

vi.mock('../../context/dashboard-state-context', () => ({
  useDashboardState: () => ({
    selectedSourceNumber: 1,
    selectedSourceName: 'Lena',
    selectedAccelerator: 'gpu',
    selectedResolution: 'original',
    activeFilters: dashboardFiltersRef.current,
    processorFilterEpoch: 0,
    setSelectedSource: vi.fn(),
    setAccelerator: vi.fn(),
    setResolution: vi.fn(),
    setActiveFilters: vi.fn(),
  }),
}));

// Mock HTMLCanvasElement.getContext for all tests
beforeEach(() => {
  const mockContext = {
    willReadFrequently: true,
    clearRect: vi.fn(),
    drawImage: vi.fn(),
  };

  HTMLCanvasElement.prototype.getContext = vi.fn(() => mockContext);
  dashboardFiltersRef.current = [];
});

// Mock dependencies
vi.mock('../../hooks/useWebRTCStream', () => ({
  useWebRTCStream: vi.fn(),
}));

import { useWebRTCStream } from '@/presentation/hooks/useWebRTCStream';

describe('VideoStreamer', () => {
  const mockStartStream = vi.fn().mockResolvedValue(undefined);
  const mockStopStream = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    vi.clearAllMocks();
    (useWebRTCStream as any).mockReturnValue({
      connectionState: 'disconnected',
      isStreaming: false,
      activeSessionId: undefined,
      error: null,
      startStream: mockStartStream,
      stopStream: mockStopStream,
    });
  });

  describe('Component Rendering', () => {
    it('renders VideoSourceSelector and empty state when not streaming', () => {
      render(<VideoStreamer />);

      expect(screen.queryByTestId('filter-panel')).not.toBeInTheDocument();
      expect(screen.getByTestId('video-source-selector')).toBeInTheDocument();
      expect(screen.getByText(/No Stream Active/i)).toBeInTheDocument();
    });

    it('renders empty state when not streaming', () => {
      render(<VideoStreamer />);

      expect(screen.getByText(/No Stream Active/i)).toBeInTheDocument();
      expect(
        screen.getByText(/Select a video source to begin streaming/i)
      ).toBeInTheDocument();
    });

    it('renders VideoCanvas when streaming', () => {
      (useWebRTCStream as any).mockReturnValue({
        connectionState: 'connected',
        isStreaming: true,
        activeSessionId: 'session-123',
        error: null,
        startStream: mockStartStream,
        stopStream: mockStopStream,
      });

      render(<VideoStreamer />);

      expect(screen.getByTestId('video-canvas-container')).toBeInTheDocument();
      expect(screen.queryByText(/No Stream Active/i)).not.toBeInTheDocument();
    });
  });

  describe('Start Stream Button', () => {
    it('Start Stream button calls useWebRTCStream.startStream() per VID-01', async () => {
      render(<VideoStreamer />);

      const startButton = screen.getByRole('button', { name: /start stream/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(mockStartStream).toHaveBeenCalledWith('camera-1', []);
      });
    });

    it('Start Stream button is disabled when streaming', () => {
      (useWebRTCStream as any).mockReturnValue({
        connectionState: 'connected',
        isStreaming: true,
        activeSessionId: 'session-123',
        error: null,
        startStream: mockStartStream,
        stopStream: mockStopStream,
      });

      render(<VideoStreamer />);

      const startButton = screen.queryByRole('button', { name: /start stream/i });
      expect(startButton).not.toBeInTheDocument();
    });

    it('Start Stream button is disabled when connecting', () => {
      (useWebRTCStream as any).mockReturnValue({
        connectionState: 'connecting',
        isStreaming: false,
        activeSessionId: undefined,
        error: null,
        startStream: mockStartStream,
        stopStream: mockStopStream,
      });

      render(<VideoStreamer />);

      const startButton = screen.getByRole('button', { name: /Connecting\.\.\./i });
      expect(startButton).toBeDisabled();
    });
  });

  describe('Stop Stream Button', () => {
    it('Stop Stream button calls useWebRTCStream.stopStream() per VID-04', async () => {
      (useWebRTCStream as any).mockReturnValue({
        connectionState: 'connected',
        isStreaming: true,
        activeSessionId: 'session-123',
        error: null,
        startStream: mockStartStream,
        stopStream: mockStopStream,
      });

      render(<VideoStreamer />);

      const stopButton = screen.getByRole('button', { name: /stop stream/i });
      fireEvent.click(stopButton);

      await waitFor(() => {
        expect(mockStopStream).toHaveBeenCalled();
      });
    });

    it('Stop Stream button is enabled only when streaming', () => {
      (useWebRTCStream as any).mockReturnValue({
        connectionState: 'connected',
        isStreaming: true,
        activeSessionId: 'session-123',
        error: null,
        startStream: mockStartStream,
        stopStream: mockStopStream,
      });

      render(<VideoStreamer />);

      const stopButton = screen.getByRole('button', { name: /stop stream/i });
      expect(stopButton).toBeInTheDocument();
      expect(stopButton).not.toBeDisabled();
    });
  });

  describe('Filter Integration', () => {
    it('passes dashboard activeFilters to startStream() per D-15', async () => {
      dashboardFiltersRef.current = [{ id: 'filter1', parameters: {} }];
      render(<VideoStreamer />);

      const startButton = screen.getByRole('button', { name: /start stream/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(mockStartStream).toHaveBeenCalledWith('camera-1', [
          { id: 'filter1', parameters: {} },
        ]);
      });
    });
  });

  describe('Source Integration', () => {
    it('selected source from VideoSourceSelector is passed to startStream()', async () => {
      render(<VideoStreamer />);

      const startButton = screen.getByRole('button', { name: /start stream/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(mockStartStream).toHaveBeenCalledWith('camera-1', []);
      });
    });
  });

  describe('Connection State', () => {
    it('connection state changes button states and shows loading/error', () => {
      (useWebRTCStream as any).mockReturnValue({
        connectionState: 'connecting',
        isStreaming: false,
        activeSessionId: undefined,
        error: null,
        startStream: mockStartStream,
        stopStream: mockStopStream,
      });

      render(<VideoStreamer />);

      const startButton = screen.getByRole('button', { name: /Connecting\.\.\./i });
      expect(startButton).toBeDisabled();
      expect(screen.queryByRole('button', { name: /stop stream/i })).not.toBeInTheDocument();
    });

    it('shows error message when connectionState is failed', () => {
      const mockError = new Error('Connection failed');
      (useWebRTCStream as any).mockReturnValue({
        connectionState: 'failed',
        isStreaming: false,
        activeSessionId: undefined,
        error: mockError,
        startStream: mockStartStream,
        stopStream: mockStopStream,
      });

      render(<VideoStreamer />);

      expect(screen.getByText(/Connection failed/i)).toBeInTheDocument();
    });
  });

  describe('Copywriting and Styling', () => {
    it('uses UI-SPEC copywriting', () => {
      render(<VideoStreamer />);

      expect(screen.getByText(/Start Stream/i)).toBeInTheDocument();
      expect(screen.getByText(/No Stream Active/i)).toBeInTheDocument();
      expect(
        screen.getByText(/Select a video source to begin streaming/i)
      ).toBeInTheDocument();
    });

    it('uses UI-SPEC colors', () => {
      (useWebRTCStream as any).mockReturnValue({
        connectionState: 'connected',
        isStreaming: true,
        activeSessionId: 'session-123',
        error: null,
        startStream: mockStartStream,
        stopStream: mockStopStream,
      });

      render(<VideoStreamer />);

      const stopButton = screen.getByRole('button', { name: /stop stream/i });
      expect(stopButton.className).toContain('stopButton');
    });
  });
});
