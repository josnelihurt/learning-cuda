import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import App from './App';
import { ToastProvider } from './context/toast-context';
import { GrpcClientsProvider } from './providers/grpc-clients-provider';
import { VideoStreamer } from './components/video/VideoStreamer';
import { HealthIndicator } from './components/health/HealthIndicator';

// Mock the VideoStreamer component to avoid complex dependency setup
vi.mock('./components/video/VideoStreamer', () => ({
  VideoStreamer: () => <div data-testid="video-streamer">VideoStreamer</div>,
}));

// Mock the HealthIndicator component
vi.mock('./components/health/HealthIndicator', () => ({
  HealthIndicator: ({ isHealthy, loading }: { isHealthy: boolean; loading: boolean }) => (
    <div data-testid="health-indicator" data-healthy={isHealthy} data-loading={loading}>
      HealthIndicator
    </div>
  ),
}));

// Mock useHealthMonitor hook
vi.mock('./hooks/useHealthMonitor', () => ({
  useHealthMonitor: () => ({
    isHealthy: true,
    loading: false,
  }),
}));

describe('App', () => {
  beforeEach(() => {
    // Mock console.error to detect errors during rendering
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('should render VideoStreamer component', () => {
    render(<App />);
    const videoStreamer = screen.getByTestId('video-streamer');
    expect(videoStreamer).toBeInTheDocument();
  });

  it('should render HealthIndicator in header', () => {
    render(<App />);
    const healthIndicator = screen.getByTestId('health-indicator');
    expect(healthIndicator).toBeInTheDocument();
  });

  it('should render without console errors', () => {
    render(<App />);
    expect(console.error).not.toHaveBeenCalled();
  });

  it('should have proper structure with header and main content', () => {
    const { container } = render(<App />);

    // Check for header with navbar
    const navbar = container.querySelector('.navbar');
    expect(navbar).toBeInTheDocument();

    // Check for main content area
    const mainContent = container.querySelector('.main-content');
    expect(mainContent).toBeInTheDocument();

    // Check that VideoStreamer is in main content
    const videoStreamer = screen.getByTestId('video-streamer');
    expect(videoStreamer).toBeInTheDocument();
  });

  it('should wrap content in providers (ToastProvider and GrpcClientsProvider are in main.tsx)', () => {
    // This test verifies that App can render when wrapped in the expected providers
    render(
      <ToastProvider>
        <GrpcClientsProvider>
          <App />
        </GrpcClientsProvider>
      </ToastProvider>
    );

    const videoStreamer = screen.getByTestId('video-streamer');
    expect(videoStreamer).toBeInTheDocument();
  });
});
