import { useState, useCallback, useEffect } from 'react';
import { manageWebRTC } from '../infrastructure/connection/webrtc-manage';
import { useToast } from './useToast';
import type { ActiveFilterState } from '../components/filters/FilterPanel';

type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'failed';

interface WebRTCStreamState {
  connectionState: ConnectionState;
  isStreaming: boolean;
  activeSessionId: string | null;
  error: Error | null;
}

export function useWebRTCStream() {
  const [state, setState] = useState<WebRTCStreamState>({
    connectionState: 'disconnected',
    isStreaming: false,
    activeSessionId: null,
    error: null,
  });

  const { error: showError } = useToast();

  const startStream = useCallback(async (sourceId: string, filters: ActiveFilterState[]) => {
    try {
      // Validate sourceId per threat model T-04-06
      if (!sourceId || sourceId.trim() === '') {
        throw new Error('Source ID cannot be empty');
      }

      // Update state to connecting
      setState(prev => ({
        ...prev,
        connectionState: 'connecting',
        error: null,
      }));

      // Note: filters are stored for future use (filter update requires restart per D-15)
      // For now, we just create the session with sourceId
      // Filters will be applied via signaling in future implementation

      const session = await manageWebRTC.createSession(sourceId);

      if (!session) {
        throw new Error('Failed to create WebRTC session');
      }

      // Update state to connected
      setState(prev => ({
        ...prev,
        connectionState: 'connected',
        isStreaming: true,
        activeSessionId: session.getId(),
      }));
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));

      // Update state to failed
      setState(prev => ({
        ...prev,
        connectionState: 'failed',
        isStreaming: false,
        activeSessionId: null,
        error,
      }));

      // Show toast notification on failure per D-12
      showError('Connection Failed', error.message);
    }
  }, [showError]);

  const stopStream = useCallback(async () => {
    const { activeSessionId } = state;

    if (activeSessionId) {
      try {
        await manageWebRTC.closeSession(activeSessionId);
      } catch (err) {
        // Log error but don't show toast (cleanup failure is non-critical)
        console.error('Failed to close session:', err);
      }
    }

    // Reset state to disconnected
    setState({
      connectionState: 'disconnected',
      isStreaming: false,
      activeSessionId: null,
      error: null,
    });
  }, [state.activeSessionId]);

  // Cleanup on unmount per D-13
  useEffect(() => {
    return () => {
      if (state.isStreaming && state.activeSessionId) {
        const closePromise = manageWebRTC.closeSession(state.activeSessionId);
        if (closePromise && typeof closePromise.catch === 'function') {
          closePromise.catch(err => {
            console.error('Failed to cleanup session on unmount:', err);
          });
        }
      }
    };
  }, [state.isStreaming, state.activeSessionId]);

  return {
    ...state,
    startStream,
    stopStream,
  };
}
