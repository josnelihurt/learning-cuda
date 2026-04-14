import { useEffect, useMemo, useState } from 'react';
import { createPromiseClient } from '@connectrpc/connect';
import { ConfigService } from '@/gen/config_service_connect';
import type { ToolCategory } from '@/gen/config_service_pb';
import { createGrpcConnectTransport } from '@/infrastructure/grpc/create-grpc-transport';
import { useAppServices } from '../../providers/app-services-provider';
import { useToast } from '../../hooks/useToast';

declare const __APP_VERSION__: string;

type ReactNavbarControlsProps = {
  onOpenFeatureFlags: () => void;
};

type VersionField = {
  label: string;
  value: string;
};

export function ReactNavbarControls({ onOpenFeatureFlags }: ReactNavbarControlsProps) {
  const [isToolsOpen, setIsToolsOpen] = useState(false);
  const [isVersionOpen, setIsVersionOpen] = useState(false);
  const [versionFields, setVersionFields] = useState<VersionField[]>([]);
  const [environment, setEnvironment] = useState('Loading...');
  const [isVersionLoading, setIsVersionLoading] = useState(true);
  const { container } = useAppServices();
  const toast = useToast();
  const transport = useMemo(() => createGrpcConnectTransport(), []);

  useEffect(() => {
    const handleDocumentClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (!target) {
        return;
      }
      if (!target.closest('[data-testid="react-tools-dropdown"]')) {
        setIsToolsOpen(false);
      }
      if (!target.closest('[data-testid="react-version-tooltip"]')) {
        setIsVersionOpen(false);
      }
    };
    document.addEventListener('click', handleDocumentClick);
    return () => {
      document.removeEventListener('click', handleDocumentClick);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadSystemInfo = async () => {
      try {
        const client = createPromiseClient(ConfigService, transport);
        const systemInfo = await client.getSystemInfo({});
        if (cancelled) {
          return;
        }
        const version = systemInfo.version;
        const fields: VersionField[] = [];
        if (version?.goVersion) fields.push({ label: 'Go Version', value: version.goVersion });
        if (version?.cppVersion) fields.push({ label: 'C++ Version', value: version.cppVersion });
        if (version?.protoVersion) fields.push({ label: 'Proto Version', value: version.protoVersion });
        if (version?.branch) fields.push({ label: 'Branch', value: version.branch });
        if (version?.buildTime) {
          fields.push({ label: 'Build Time', value: new Date(version.buildTime).toLocaleString() });
        }
        if (version?.commitHash) fields.push({ label: 'Commit Hash', value: version.commitHash });
        setVersionFields(fields);
        setEnvironment(systemInfo.environment || 'Unknown');
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        toast.error('Version Error', message);
        setEnvironment('Error');
      } finally {
        if (!cancelled) {
          setIsVersionLoading(false);
        }
      }
    };
    void loadSystemInfo();
    return () => {
      cancelled = true;
    };
  }, [toast, transport]);

  const toolCategories =
    typeof container.getToolsService === 'function'
      ? container.getToolsService().getCategories()
      : [];

  return (
    <>
      <div style={{ position: 'relative', display: 'inline-block' }} data-testid="react-tools-dropdown">
        <button
          type="button"
          className="feature-flags-btn"
          onClick={() => setIsToolsOpen((value) => !value)}
          data-testid="tools-dropdown-button"
        >
          <span>Tools</span>
          <span>{isToolsOpen ? '▲' : '▼'}</span>
        </button>
        {isToolsOpen ? (
          <div
            style={{
              position: 'absolute',
              top: 'calc(100% + 8px)',
              right: 0,
              background: 'rgba(30, 30, 40, 0.98)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: '8px',
              minWidth: '220px',
              zIndex: 1000,
            }}
          >
            {toolCategories.flatMap((category: ToolCategory) => category.tools).map((tool) => (
              <button
                key={tool.name}
                type="button"
                className="feature-flags-btn"
                style={{
                  width: '100%',
                  border: 'none',
                  borderRadius: 0,
                  justifyContent: 'flex-start',
                  padding: '10px 14px',
                  transform: 'none',
                }}
                onClick={() => {
                  setIsToolsOpen(false);
                  if (tool.type === 'url' && tool.url) {
                    window.open(tool.url, '_blank', 'noopener,noreferrer');
                    return;
                  }
                  if (tool.type === 'action' && tool.action === 'sync_flags') {
                    void (async () => {
                      try {
                        const client = createPromiseClient(ConfigService, transport);
                        const response = await client.syncFeatureFlags({});
                        toast.success('Feature Flags', response.message);
                      } catch (error) {
                        const message = error instanceof Error ? error.message : String(error);
                        toast.error('Sync Error', message);
                      }
                    })();
                  }
                }}
              >
                {tool.name}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <button
        type="button"
        className="feature-flags-btn"
        onClick={onOpenFeatureFlags}
        data-testid="feature-flags-button"
      >
        Feature Flags
      </button>

      <button type="button" className="feature-flags-btn" style={{ display: 'none' }}>
        Sync Flags
      </button>

      <div style={{ position: 'relative', display: 'inline-block' }} data-testid="react-version-tooltip">
        <button
          type="button"
          className="info-btn"
          title="Version Information"
          onClick={() => setIsVersionOpen((value) => !value)}
        >
          <span>i</span>
        </button>
        {isVersionOpen ? (
          <div
            style={{
              position: 'absolute',
              top: 'calc(100% + 10px)',
              right: 0,
              background: '#fff',
              border: '1px solid #ccc',
              borderRadius: '8px',
              padding: '16px',
              minWidth: '320px',
              zIndex: 1001,
              color: '#111',
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: '8px' }}>Version Information</div>
            {isVersionLoading ? <div>Loading...</div> : null}
            {versionFields.map((field) => (
              <div key={field.label} style={{ display: 'flex', gap: '8px', fontSize: '12px' }}>
                <span style={{ fontWeight: 700, minWidth: '120px' }}>{field.label}:</span>
                <span>{field.value}</span>
              </div>
            ))}
            <div style={{ marginTop: '8px', borderTop: '1px solid #eee', paddingTop: '8px', fontSize: '12px' }}>
              <span style={{ fontWeight: 700, minWidth: '120px', display: 'inline-block' }}>
                Environment:
              </span>
              <span>{environment}</span>
            </div>
            <div style={{ marginTop: '8px', fontSize: '11px', color: '#666' }}>Build: {__APP_VERSION__}</div>
          </div>
        ) : null}
      </div>
    </>
  );
}
