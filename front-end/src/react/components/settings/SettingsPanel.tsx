import { useState, useMemo } from 'react';
import { GetStreamConfigResponse, StreamEndpoint } from '@/gen/config_service_pb';
import { useConfig } from '../../hooks/useConfig';
import { useToast } from '../../hooks/useToast';
import styles from './SettingsPanel.module.css';

export function SettingsPanel() {
  const { config, loading, saving, error, fetchConfig, updateConfig } = useConfig();
  const toast = useToast();

  // Local state for editing
  const [localConfig, setLocalConfig] = useState<StreamEndpoint[]>([]);

  // Sync local config when config loads
  useState(() => {
    if (config?.endpoints) {
      setLocalConfig([...config.endpoints]);
    }
  });

  // Check for changes
  const hasChanges = useMemo(() => {
    if (!config?.endpoints) return false;
    if (localConfig.length !== config.endpoints.length) return true;
    return localConfig.some((ep, i) => {
      const configEp = config.endpoints[i];
      return (
        ep.type !== configEp.type ||
        ep.endpoint !== configEp.endpoint ||
        ep.transportFormat !== configEp.transportFormat ||
        ep.logLevel !== configEp.logLevel ||
        ep.consoleLogging !== configEp.consoleLogging
      );
    });
  }, [config, localConfig]);

  // Validation errors
  const validationErrors = useMemo(() => {
    const errors: Record<number, string> = {};
    localConfig.forEach((ep, index) => {
      if (!ep.endpoint.trim()) {
        errors[index] = 'Endpoint URL is required';
      } else if (!isValidUrl(ep.endpoint)) {
        errors[index] = 'Invalid URL format';
      }
    });
    return errors;
  }, [localConfig]);

  const handleSave = async () => {
    if (Object.keys(validationErrors).length > 0) {
      toast.error('Validation Error', 'Please fix the errors before saving');
      return;
    }

    if (!config) return;

    const newConfig = new GetStreamConfigResponse({
      endpoints: localConfig,
      traceContext: config.traceContext,
    });

    await updateConfig(newConfig);
  };

  const handleDiscard = () => {
    if (config?.endpoints) {
      setLocalConfig([...config.endpoints]);
      toast.info('Changes Discarded', 'Your unsaved changes have been discarded');
    }
  };

  const updateEndpoint = (index: number, field: keyof StreamEndpoint, value: string | boolean) => {
    const updated = [...localConfig];
    updated[index] = new StreamEndpoint({
      ...updated[index],
      [field]: value,
    });
    setLocalConfig(updated);
  };

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading configuration...</div>
      </div>
    );
  }

  if (error && !config) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>
          <h3>Error Loading Configuration</h3>
          <p>{error}</p>
          <button onClick={fetchConfig} className={styles.button}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  // TODO: Remove this read-only message when updateStreamConfig RPC is available
  const hasUpdateRpc = false; // Set to true when backend supports update

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2>System Configuration</h2>
        <button onClick={fetchConfig} className={styles.iconButton} title="Refresh">
          ↻
        </button>
      </div>

      {hasUpdateRpc ? (
        <>
          <div className={styles.section}>
            <h3>Stream Endpoints</h3>
            {localConfig.map((endpoint, index) => (
              <div key={index} className={styles.endpointCard}>
                <div className={styles.formGroup}>
                  <label>Type</label>
                  <select
                    value={endpoint.type}
                    onChange={(e) => updateEndpoint(index, 'type', e.target.value)}
                    className={styles.select}
                  >
                    <option value="gRPC">gRPC</option>
                    <option value="WebSocket">WebSocket</option>
                  </select>
                </div>

                <div className={styles.formGroup}>
                  <label>Endpoint URL *</label>
                  <input
                    type="text"
                    value={endpoint.endpoint}
                    onChange={(e) => updateEndpoint(index, 'endpoint', e.target.value)}
                    className={`${styles.input} ${validationErrors[index] ? styles.inputError : ''}`}
                    placeholder="e.g., /ws or grpc://localhost:50051"
                  />
                  {validationErrors[index] && (
                    <span className={styles.validationError}>{validationErrors[index]}</span>
                  )}
                </div>

                <div className={styles.formGroup}>
                  <label>Transport Format</label>
                  <select
                    value={endpoint.transportFormat}
                    onChange={(e) => updateEndpoint(index, 'transportFormat', e.target.value)}
                    className={styles.select}
                  >
                    <option value="raw">Raw</option>
                    <option value="binary">Binary</option>
                  </select>
                </div>

                <div className={styles.formGroup}>
                  <label>Log Level</label>
                  <select
                    value={endpoint.logLevel}
                    onChange={(e) => updateEndpoint(index, 'logLevel', e.target.value)}
                    className={styles.select}
                  >
                    <option value="DEBUG">DEBUG</option>
                    <option value="INFO">INFO</option>
                    <option value="WARN">WARN</option>
                    <option value="ERROR">ERROR</option>
                  </select>
                </div>

                <div className={styles.formGroup}>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={endpoint.consoleLogging}
                      onChange={(e) => updateEndpoint(index, 'consoleLogging', e.target.checked)}
                      className={styles.checkbox}
                    />
                    <span>Console Logging</span>
                  </label>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.actions}>
            <button
              onClick={handleDiscard}
              disabled={!hasChanges || saving}
              className={`${styles.button} ${styles.secondaryButton}`}
            >
              Discard Changes
            </button>
            <button
              onClick={handleSave}
              disabled={!hasChanges || saving || Object.keys(validationErrors).length > 0}
              className={`${styles.button} ${styles.primaryButton}`}
            >
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </>
      ) : (
        <div className={styles.readonly}>
          <h3>Configuration is Read-Only</h3>
          <p>The updateStreamConfig RPC is not yet available in the backend.</p>
          <p>Current configuration is displayed below:</p>
          {config?.endpoints && config.endpoints.length > 0 ? (
            <div className={styles.section}>
              {config.endpoints.map((endpoint, index) => (
                <div key={index} className={styles.endpointCard}>
                  <div className={styles.field}>
                    <strong>Type:</strong> {endpoint.type}
                  </div>
                  <div className={styles.field}>
                    <strong>Endpoint:</strong> {endpoint.endpoint}
                  </div>
                  <div className={styles.field}>
                    <strong>Transport Format:</strong> {endpoint.transportFormat}
                  </div>
                  <div className={styles.field}>
                    <strong>Log Level:</strong> {endpoint.logLevel}
                  </div>
                  <div className={styles.field}>
                    <strong>Console Logging:</strong> {endpoint.consoleLogging ? 'Enabled' : 'Disabled'}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p>No endpoints configured.</p>
          )}
        </div>
      )}
    </div>
  );
}

// Simple URL validation
function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    // Allow relative URLs starting with /
    if (url.startsWith('/')) {
      return true;
    }
    return false;
  }
}
