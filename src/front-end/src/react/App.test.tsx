import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, cleanup } from '@testing-library/react';
import App from './App';
import { ToastProvider } from './context/toast-context';
import { GrpcClientsProvider } from './providers/grpc-clients-provider';

vi.mock('./providers/app-services-provider', () => ({
  useAppServices: () => ({
    container: {},
    ready: true,
  }),
}));

vi.mock('./components/video/VideoStreamer', () => ({
  VideoStreamer: () => <div data-testid="video-streamer">VideoStreamer</div>,
}));

describe('App', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('should render VideoStreamer when services are ready', () => {
    render(<App />);
    const videoStreamer = screen.getByTestId('video-streamer');
    expect(videoStreamer).toBeInTheDocument();
  });

  it('should show React load marker text', () => {
    render(<App />);
    expect(screen.getByText('React app loaded')).toBeInTheDocument();
  });

  it('should render without console errors', () => {
    render(<App />);
    expect(console.error).not.toHaveBeenCalled();
  });

  it('should have Lit-matching shell structure', () => {
    const { container } = render(<App />);

    expect(container.querySelector('.navbar')).toBeInTheDocument();
    expect(container.querySelector('.sidebar')).toBeInTheDocument();
    expect(container.querySelector('.main-content')).toBeInTheDocument();

    const videoStreamer = screen.getByTestId('video-streamer');
    expect(container.querySelector('.main-content')?.contains(videoStreamer)).toBe(true);
  });

  it('should render when wrapped like main.tsx', () => {
    render(
      <ToastProvider>
        <GrpcClientsProvider>
          <App />
        </GrpcClientsProvider>
      </ToastProvider>
    );

    expect(screen.getByTestId('video-streamer')).toBeInTheDocument();
  });
});
