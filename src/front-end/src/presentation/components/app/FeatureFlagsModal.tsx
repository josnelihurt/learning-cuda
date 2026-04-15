import React, { useEffect, useMemo, useState } from 'react';
import { createPromiseClient } from '@connectrpc/connect';
import { ConfigService } from '@/gen/config_service_connect';
import { createGrpcConnectTransport } from '@/infrastructure/grpc/create-grpc-transport';
import { useToast } from '@/presentation/hooks/useToast';
import { OpenFeature } from '@openfeature/web-sdk';
import { GoFeatureFlagWebProvider } from '@openfeature/go-feature-flag-web-provider';
import './FeatureFlagsModal.css';

type FeatureFlagsModalProps = {
  isOpen: boolean;
  onClose: () => void;
};

type ManagedFlag = {
  key: string;
  name: string;
  type: string;
  enabled: boolean;
  defaultValue: string;
  description: string;
};

export function FeatureFlagsModal({ isOpen, onClose }: FeatureFlagsModalProps) {
  const [loading, setLoading] = useState(false);
  const [savingKey, setSavingKey] = useState<string | null>(null);
  const [flags, setFlags] = useState<ManagedFlag[]>([]);
  const [openFeatureReady, setOpenFeatureReady] = useState(false);
  const toast = useToast();
  const client = useMemo(
    () => createPromiseClient(ConfigService, createGrpcConnectTransport()),
    []
  );

  const loadFlags = async () => {
    setLoading(true);
    try {
      const response = await client.listFeatureFlags({});
      setFlags(
        response.flags.map((flag) => ({
          key: flag.key,
          name: flag.name,
          type: flag.type,
          enabled: flag.enabled,
          defaultValue: flag.defaultValue,
          description: flag.description,
        }))
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      toast.error('Feature Flags', message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    void loadFlags();
    void (async () => {
      try {
        await OpenFeature.setContext({ targetingKey: 'frontend-default' });
        await OpenFeature.setProvider(
          new GoFeatureFlagWebProvider({
            endpoint: `${window.location.origin}`,
            apiTimeout: 5000,
          })
        );
        setOpenFeatureReady(true);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        toast.error('OpenFeature', `Provider init failed: ${message}`);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const updateFlag = async (flag: ManagedFlag) => {
    setSavingKey(flag.key);
    try {
      await client.upsertFeatureFlag({
        flag: {
          key: flag.key,
          name: flag.name,
          type: flag.type,
          enabled: flag.enabled,
          defaultValue: flag.defaultValue,
          description: flag.description,
        },
      });
      toast.success('Feature Flags', `Flag ${flag.key} updated`);
      await loadFlags();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      toast.error('Feature Flags', message);
    } finally {
      setSavingKey(null);
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="feature-flags-modal-overlay">
      <div className="feature-flags-modal-backdrop" onClick={onClose} />
      <div className="feature-flags-modal-panel">
        <div className="feature-flags-modal-header">
          <h2 className="feature-flags-modal-title">Feature Flags</h2>
          <div className="feature-flags-modal-header-actions">
            <button type="button" className="feature-flags-btn" disabled={loading} onClick={() => void loadFlags()}>
              Refresh
            </button>
            <button type="button" className="feature-flags-btn" onClick={onClose}>
              ×
            </button>
          </div>
        </div>
        <div className="feature-flags-modal-content">
          <div className="feature-flags-modal-provider-status">
            Front-end provider: {openFeatureReady ? 'ready' : 'fallback to backend'}
          </div>
          {loading ? (
            <div>Loading flags...</div>
          ) : (
            <table className="feature-flags-modal-table">
              <thead>
                <tr>
                  <th className="feature-flags-modal-cell feature-flags-modal-header-cell">Key</th>
                  <th className="feature-flags-modal-cell feature-flags-modal-header-cell">Type</th>
                  <th className="feature-flags-modal-cell feature-flags-modal-header-cell">Enabled</th>
                  <th className="feature-flags-modal-cell feature-flags-modal-header-cell">Default</th>
                  <th className="feature-flags-modal-cell feature-flags-modal-header-cell">Actions</th>
                </tr>
              </thead>
              <tbody>
                {flags.map((flag) => (
                  <tr key={flag.key}>
                    <td className="feature-flags-modal-cell">{flag.key}</td>
                    <td className="feature-flags-modal-cell">{flag.type}</td>
                    <td className="feature-flags-modal-cell">
                      <input
                        type="checkbox"
                        checked={flag.enabled}
                        onChange={(event) => {
                          setFlags((prev) =>
                            prev.map((item) =>
                              item.key === flag.key ? { ...item, enabled: event.target.checked } : item
                            )
                          );
                        }}
                      />
                    </td>
                    <td className="feature-flags-modal-cell">
                      <input
                        value={flag.defaultValue}
                        onChange={(event) => {
                          setFlags((prev) =>
                            prev.map((item) =>
                              item.key === flag.key ? { ...item, defaultValue: event.target.value } : item
                            )
                          );
                        }}
                      />
                    </td>
                    <td className="feature-flags-modal-cell">
                      <button
                        type="button"
                        className="feature-flags-btn"
                        disabled={savingKey === flag.key}
                        onClick={() => void updateFlag(flag)}
                      >
                        {savingKey === flag.key ? 'Saving...' : 'Save'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
