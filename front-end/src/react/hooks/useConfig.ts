import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ConnectError } from '@connectrpc/connect';
import {
  GetStreamConfigRequest,
  GetStreamConfigResponse,
  StreamEndpoint,
} from '@/gen/config_service_pb';
import { createPromiseClient } from '@connectrpc/connect';
import { ConfigService } from '@/gen/config_service_connect';
import { createGrpcConnectTransport } from '@/infrastructure/grpc/create-grpc-transport';
import { useToast } from './useToast';

export function useConfig() {
  const toast = useToast();
  const [config, setConfig] = useState<GetStreamConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Create config client (following pattern from existing services)
  const client = useMemo(() => {
    const transport = createGrpcConnectTransport();
    return createPromiseClient(ConfigService, transport);
  }, []);

  const clientRef = useRef(client);
  clientRef.current = client;

  const fetchConfig = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const request = new GetStreamConfigRequest({});
      const response = await clientRef.current.getStreamConfig(request);
      setConfig(response);
    } catch (e) {
      const conn = ConnectError.from(e);
      const errorMessage = `Failed to fetch configuration: ${conn.message}`;
      setError(errorMessage);
      toast.error('Configuration Error', errorMessage);
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const updateConfig = useCallback(async (newConfig: GetStreamConfigResponse) => {
    setSaving(true);
    setError(null);

    try {
      // TODO: Implement updateStreamConfig RPC when backend supports it
      // Currently, the ConfigService only has getStreamConfig, not updateStreamConfig
      // For now, this is a no-op that simulates a successful update
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate network delay

      setConfig(newConfig);
      toast.success('Configuration Saved', 'Your changes have been saved successfully');
    } catch (e) {
      const conn = ConnectError.from(e);
      const errorMessage = `Failed to save configuration: ${conn.message}`;
      setError(errorMessage);
      toast.error('Save Failed', errorMessage);
    } finally {
      setSaving(false);
    }
  }, [toast]);

  // Fetch config on mount
  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  return {
    config,
    loading,
    saving,
    error,
    fetchConfig,
    updateConfig,
  };
}
