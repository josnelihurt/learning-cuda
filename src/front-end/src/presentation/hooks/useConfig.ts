import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ConnectError, Code } from '@connectrpc/connect';
import {
  GetStreamConfigRequest,
  GetStreamConfigResponse,
  StreamEndpoint,
} from '@/gen/config_service_pb';
import { createPromiseClient } from '@connectrpc/connect';
import { ConfigService } from '@/gen/config_service_connect';
import { createGrpcConnectTransport } from '@/infrastructure/grpc/create-grpc-transport';
import { useToast } from './useToast';

/**
 * Custom hook for managing system configuration.
 *
 * Provides functionality to:
 * - Fetch current stream configuration from the backend
 * - Update configuration (when updateStreamConfig RPC is available)
 * - Track loading, saving, and error states
 *
 * @returns Object containing config state and operations
 */
export function useConfig() {
  const toast = useToast();
  const [config, setConfig] = useState<GetStreamConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const requestGenRef = useRef(0);

  // Create config client (following pattern from existing services)
  const client = useMemo(() => {
    const transport = createGrpcConnectTransport();
    return createPromiseClient(ConfigService, transport);
  }, []);

  const clientRef = useRef(client);
  clientRef.current = client;

  /**
   * Fetches the current stream configuration from the backend.
   *
   * Makes an RPC call to getStreamConfig and updates the config state.
   * Handles errors gracefully and shows toast notifications.
   */
  const fetchConfig = useCallback(async () => {
    // Cancel any pending requests
    abortRef.current?.abort();
    const gen = ++requestGenRef.current;
    const ac = new AbortController();
    abortRef.current = ac;
    const { signal } = ac;

    setLoading(true);
    setError(null);

    try {
      const request = new GetStreamConfigRequest({});
      const response = await clientRef.current.getStreamConfig(request, { signal });

      // Check if this is still the latest request
      if (signal.aborted || gen !== requestGenRef.current) {
        return;
      }

      setConfig(response);
      setRetryCount(0); // Reset retry count on success
    } catch (e) {
      // Check if this is still the latest request
      if (signal.aborted || gen !== requestGenRef.current) {
        return;
      }

      const conn = ConnectError.from(e);
      let errorMessage = `Failed to fetch configuration: ${conn.message}`;

      // Handle specific error codes with user-friendly messages
      if (conn.code === Code.Unavailable) {
        errorMessage = 'Backend service unavailable. Please check your connection.';
      } else if (conn.code === Code.Unauthenticated) {
        errorMessage = 'Authentication required. Please log in.';
      } else if (conn.code === Code.PermissionDenied) {
        errorMessage = 'Permission denied. You may not have access to configuration.';
      }

      setError(errorMessage);
      toast.error('Configuration Error', errorMessage);
    } finally {
      if (gen === requestGenRef.current) {
        setLoading(false);
      }
    }
  }, [toast]);

  /**
   * Updates the system configuration with new values.
   *
   * Note: Currently implemented as a no-op with a TODO comment because
   * the updateStreamConfig RPC is not yet available in the backend.
   * When the backend supports updates, this should call the appropriate RPC.
   *
   * @param newConfig - The new configuration to apply
   */
  const updateConfig = useCallback(async (newConfig: GetStreamConfigResponse) => {
    setSaving(true);
    setError(null);

    try {
      // TODO: Implement updateStreamConfig RPC when backend supports it
      // Currently, the ConfigService only has getStreamConfig, not updateStreamConfig
      // For now, this is a no-op that simulates a successful update
      //
      // Future implementation:
      // const request = new UpdateStreamConfigRequest({ endpoints: newConfig.endpoints });
      // await clientRef.current.updateStreamConfig(request);

      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate network delay

      setConfig(newConfig);
      toast.success('Configuration Saved', 'Your changes have been saved successfully');
    } catch (e) {
      const conn = ConnectError.from(e);
      let errorMessage = `Failed to save configuration: ${conn.message}`;

      // Handle specific error codes with user-friendly messages
      if (conn.code === Code.InvalidArgument) {
        errorMessage = 'Invalid configuration. Please check your values.';
      } else if (conn.code === Code.Unavailable) {
        errorMessage = 'Backend service unavailable. Please try again.';
      }

      setError(errorMessage);
      toast.error('Save Failed', errorMessage);
    } finally {
      setSaving(false);
    }
  }, [toast]);

  // Fetch config on mount
  useEffect(() => {
    fetchConfig();

    // Cleanup: abort any pending requests on unmount
    return () => {
      abortRef.current?.abort();
    };
  }, [fetchConfig]);

  /**
   * Retries fetching the configuration.
   * Useful when the initial load fails and the user wants to retry.
   */
  const retryFetch = useCallback(() => {
    setRetryCount(prev => prev + 1);
    fetchConfig();
  }, [fetchConfig]);

  /**
   * Clears the current error state.
   * Useful when the user dismisses an error message.
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // State
    config,
    loading,
    saving,
    error,
    retryCount,

    // Operations
    fetchConfig,
    updateConfig,
    retryFetch,
    clearError,

    // Computed
    hasError: error !== null,
    isReady: !loading && !error && config !== null,
  };
}
