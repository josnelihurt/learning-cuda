import { useMemo, useState } from 'react';
import { createPromiseClient } from '@connectrpc/connect';
import { ConfigService } from '@/gen/config_service_connect';
import { createGrpcConnectTransport } from '@/infrastructure/grpc/create-grpc-transport';
import { useToast } from '../../hooks/useToast';

type FeatureFlagsModalProps = {
  isOpen: boolean;
  onClose: () => void;
};

export function FeatureFlagsModal({ isOpen, onClose }: FeatureFlagsModalProps) {
  const [syncing, setSyncing] = useState(false);
  const toast = useToast();
  const client = useMemo(
    () => createPromiseClient(ConfigService, createGrpcConnectTransport()),
    []
  );

  if (!isOpen) {
    return null;
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 10000 }}>
      <div style={{ position: 'absolute', inset: 0, background: 'rgba(0, 0, 0, 0.6)' }} onClick={onClose} />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '90vw',
          height: '85vh',
          maxWidth: '1198px',
          maxHeight: '800px',
          background: 'rgba(20, 20, 28, 0.95)',
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div style={{ padding: '12px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ margin: 0, fontSize: '16px' }}>Feature Flags</h2>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              type="button"
              className="feature-flags-btn"
              disabled={syncing}
              onClick={() => {
                if (syncing) {
                  return;
                }
                setSyncing(true);
                void client
                  .syncFeatureFlags({})
                  .then((response) => {
                    toast.success('Feature Flags', response.message);
                  })
                  .catch((error) => {
                    const message = error instanceof Error ? error.message : String(error);
                    toast.error('Sync Error', message);
                  })
                  .finally(() => {
                    setSyncing(false);
                  });
              }}
            >
              {syncing ? 'Syncing...' : 'Sync'}
            </button>
            <button type="button" className="feature-flags-btn" onClick={onClose}>
              ×
            </button>
          </div>
        </div>
        <iframe
          title="Flipt Feature Flags"
          src={`${window.location.origin}/flipt/#/namespaces/default/flags`}
          style={{ flex: 1, border: 'none', borderRadius: '0 0 8px 8px', background: 'white' }}
        />
      </div>
    </div>
  );
}
