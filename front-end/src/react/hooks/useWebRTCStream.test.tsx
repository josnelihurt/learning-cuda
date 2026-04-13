import { renderHook, act, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useWebRTCStream } from './useWebRTCStream';
import { manageWebRTC } from '../infrastructure/connection/webrtc-manage';
import type { ActiveFilterState } from '../components/filters/FilterPanel';

// Mock manageWebRTC
vi.mock('../infrastructure/connection/webrtc-manage', () => ({
  manageWebRTC: {
    createSession: vi.fn(),
    closeSession: vi.fn(),
  },
}));

// Mock useToast
vi.mock('./useToast', () => ({
  useToast: vi.fn(() => ({
    error: vi.fn(),
  })),
}));

describe('useWebRTCStream', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should expose connectionState (connecting, connected, disconnected, failed)', () => {
    const { result } = renderHook(() => useWebRTCStream());

    expect(result.current.connectionState).toBe('disconnected');
    expect(typeof result.current.connectionState).toBe('string');
  });

  it('should expose isStreaming boolean (true when connected)', () => {
    const { result } = renderHook(() => useWebRTCStream());

    expect(result.current.isStreaming).toBe(false);
    expect(typeof result.current.isStreaming).toBe('boolean');
  });

  it('should expose activeSessionId (null when not streaming)', () => {
    const { result } = renderHook(() => useWebRTCStream());

    expect(result.current.activeSessionId).toBeNull();
  });

  it('should expose error (null when no error)', () => {
    const { result } = renderHook(() => useWebRTCStream());

    expect(result.current.error).toBeNull();
  });

  it('startStream(sourceId, filters) should call manageWebRTC.createSession()', async () => {
    const mockSession = {
      getId: () => 'session-123',
      getSourceId: () => 'source-abc',
      getCreatedAt: () => new Date(),
      getLastHeartbeat: () => new Date(),
    };
    vi.mocked(manageWebRTC.createSession).mockResolvedValue(mockSession as any);

    const { result } = renderHook(() => useWebRTCStream());

    await act(async () => {
      await result.current.startStream('source-abc', []);
    });

    expect(manageWebRTC.createSession).toHaveBeenCalledWith('source-abc');
  });

  it('startStream() should update connectionState to connecting then connected', async () => {
    const mockSession = {
      getId: () => 'session-123',
      getSourceId: () => 'source-abc',
      getCreatedAt: () => new Date(),
      getLastHeartbeat: () => new Date(),
    };
    vi.mocked(manageWebRTC.createSession).mockImplementation(
      () =>
        new Promise((resolve) => {
          // Check that state is 'connecting' during the async operation
          setTimeout(() => resolve(mockSession as any), 10);
        })
    );

    const { result } = renderHook(() => useWebRTCStream());

    await act(async () => {
      const startPromise = result.current.startStream('source-abc', []);
      // State should be 'connecting' immediately
      expect(result.current.connectionState).toBe('connecting');
      await startPromise;
    });

    // State should be 'connected' after successful completion
    expect(result.current.connectionState).toBe('connected');
  });

  it('stopStream() should call manageWebRTC.closeSession() and cleanup state', async () => {
    const mockSession = {
      getId: () => 'session-123',
      getSourceId: () => 'source-abc',
      getCreatedAt: () => new Date(),
      getLastHeartbeat: () => new Date(),
    };
    vi.mocked(manageWebRTC.createSession).mockResolvedValue(mockSession as any);

    const { result } = renderHook(() => useWebRTCStream());

    // Start stream first
    await act(async () => {
      await result.current.startStream('source-abc', []);
    });

    expect(result.current.activeSessionId).toBe('session-123');

    // Stop stream
    await act(async () => {
      await result.current.stopStream();
    });

    expect(manageWebRTC.closeSession).toHaveBeenCalledWith('session-123');
  });

  it('stopStream() should set connectionState to disconnected, activeSessionId to null', async () => {
    const mockSession = {
      getId: () => 'session-123',
      getSourceId: () => 'source-abc',
      getCreatedAt: () => new Date(),
      getLastHeartbeat: () => new Date(),
    };
    vi.mocked(manageWebRTC.createSession).mockResolvedValue(mockSession as any);

    const { result } = renderHook(() => useWebRTCStream());

    // Start stream first
    await act(async () => {
      await result.current.startStream('source-abc', []);
    });

    expect(result.current.activeSessionId).toBe('session-123');
    expect(result.current.isStreaming).toBe(true);

    // Stop stream
    await act(async () => {
      await result.current.stopStream();
    });

    expect(result.current.connectionState).toBe('disconnected');
    expect(result.current.activeSessionId).toBeNull();
    expect(result.current.isStreaming).toBe(false);
  });

  it('Hook should call useToast.error() on connection failures per D-12', async () => {
    const { useToast } = await import('./useToast');
    const mockError = vi.fn();
    vi.mocked(useToast).mockReturnValue({
      error: mockError,
      success: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
    });

    vi.mocked(manageWebRTC.createSession).mockRejectedValue(
      new Error('Connection failed')
    );

    const { result } = renderHook(() => useWebRTCStream());

    await act(async () => {
      await result.current.startStream('source-abc', []);
    });

    expect(mockError).toHaveBeenCalledWith('Connection Failed', 'Connection failed');
    expect(result.current.connectionState).toBe('failed');
    expect(result.current.error).toBeInstanceOf(Error);
  });

  it('Hook should cleanup on unmount (call stopStream if streaming)', async () => {
    const mockSession = {
      getId: () => 'session-123',
      getSourceId: () => 'source-abc',
      getCreatedAt: () => new Date(),
      getLastHeartbeat: () => new Date(),
    };
    vi.mocked(manageWebRTC.createSession).mockResolvedValue(mockSession as any);

    const { result, unmount } = renderHook(() => useWebRTCStream());

    // Start stream
    await act(async () => {
      await result.current.startStream('source-abc', []);
    });

    expect(result.current.activeSessionId).toBe('session-123');
    expect(result.current.isStreaming).toBe(true);

    // Unmount should cleanup
    unmount();

    // closeSession should have been called during cleanup
    await waitFor(() => {
      expect(manageWebRTC.closeSession).toHaveBeenCalledWith('session-123');
    });
  });

  it('should handle empty sourceId (validation per threat model T-04-06)', async () => {
    const { useToast } = await import('./useToast');
    const mockError = vi.fn();
    vi.mocked(useToast).mockReturnValue({
      error: mockError,
      success: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
    });

    vi.mocked(manageWebRTC.createSession).mockResolvedValue(null);

    const { result } = renderHook(() => useWebRTCStream());

    await act(async () => {
      await result.current.startStream('', []);
    });

    // Should fail with appropriate error
    expect(result.current.connectionState).toBe('failed');
    expect(result.current.error).toBeInstanceOf(Error);
  });

  it('should handle filters parameter (stored for future use per D-15)', async () => {
    const mockSession = {
      getId: () => 'session-123',
      getSourceId: () => 'source-abc',
      getCreatedAt: () => new Date(),
      getLastHeartbeat: () => new Date(),
    };
    vi.mocked(manageWebRTC.createSession).mockResolvedValue(mockSession as any);

    const filters: ActiveFilterState[] = [
      { id: 'filter-1', parameters: { intensity: '5' } },
      { id: 'filter-2', parameters: { threshold: '0.5' } },
    ];

    const { result } = renderHook(() => useWebRTCStream());

    await act(async () => {
      await result.current.startStream('source-abc', filters);
    });

    // Filters are accepted (not yet used in signaling, but validated to be accepted)
    expect(manageWebRTC.createSession).toHaveBeenCalledWith('source-abc');
    expect(result.current.connectionState).toBe('connected');
  });
});
