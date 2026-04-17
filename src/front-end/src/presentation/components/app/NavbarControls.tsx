import { useEffect, useMemo, useState, type ReactElement } from 'react';
import { createPromiseClient } from '@connectrpc/connect';
import { ConfigService } from '@/gen/config_service_connect';
import type { ToolCategory } from '@/gen/config_service_pb';
import { createGrpcConnectTransport } from '@/infrastructure/grpc/create-grpc-transport';
import { useAppServices } from '@/presentation/providers/app-services-provider';
import { useToast } from '@/presentation/hooks/useToast';
import styles from './NavbarControls.module.css';

declare const __APP_VERSION__: string;

type NavbarControlsProps = {
  onOpenFeatureFlags: () => void;
};

type VersionField = {
  label: string;
  value: string;
};

export function VersionTooltip(): ReactElement {
  const [isVersionOpen, setIsVersionOpen] = useState(false);
  const [versionFields, setVersionFields] = useState<VersionField[]>([]);
  const [environment, setEnvironment] = useState('Loading...');
  const [isVersionLoading, setIsVersionLoading] = useState(true);
  const toast = useToast();
  const transport = useMemo(() => createGrpcConnectTransport(), []);

  useEffect(() => {
    const handleDocumentClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (!target) {
        return;
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

  const buildVersion = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : 'dev';

  return (
    <div className={styles.versionTooltip} data-testid="react-version-tooltip">
      <button
        type="button"
        className="info-btn"
        title="Version Information"
        onClick={() => setIsVersionOpen((value) => !value)}
      >
        <span>i</span>
      </button>
      {isVersionOpen ? (
        <div className={styles.versionCard}>
          <div className={styles.tooltipHeader}>
            <span className={styles.versionTitle}>Version Information</span>
            <button
              type="button"
              className={styles.tooltipClose}
              onClick={(event) => {
                event.stopPropagation();
                setIsVersionOpen(false);
              }}
              title="Close"
            >
              ×
            </button>
          </div>
          <div className={styles.versionInfo}>
            {isVersionLoading ? <div className={styles.versionRow}>Loading...</div> : null}
            {versionFields.map((field) => (
              <div key={field.label} className={styles.versionRow}>
                <span className={styles.versionLabel}>{field.label}:</span>
                <span className={styles.versionValue}>{field.value}</span>
              </div>
            ))}
            <div className={styles.environmentRow}>
              <div className={styles.versionRow}>
                <span className={styles.versionLabel}>Environment:</span>
                <span className={styles.versionValue}>{environment}</span>
              </div>
              <div className={styles.versionRow}>
                <span className={styles.versionLabel}>Frontend:</span>
                <span className={styles.versionValue}>React app loaded</span>
              </div>
            </div>
            <div className={styles.buildInfo}>Build: {buildVersion}</div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function NavbarControls({ onOpenFeatureFlags }: NavbarControlsProps): ReactElement {
  const [isToolsOpen, setIsToolsOpen] = useState(false);
  const { container } = useAppServices();
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
    };
    document.addEventListener('click', handleDocumentClick);
    return () => {
      document.removeEventListener('click', handleDocumentClick);
    };
  }, []);

  const toolCategories =
    typeof container.getToolsService === 'function'
      ? container.getToolsService().getCategories()
      : [];

  return (
    <>
      <div className={styles.toolsDropdown} data-testid="react-tools-dropdown">
        <button
          type="button"
          className={styles.featureFlagsBtn}
          onClick={() => setIsToolsOpen((value) => !value)}
          data-testid="tools-dropdown-button"
        >
          <span>Tools</span>
          <span>{isToolsOpen ? '▲' : '▼'}</span>
        </button>
        {isToolsOpen ? (
          <div className={styles.dropdownMenu}>
            {toolCategories.flatMap((category: ToolCategory) => category.tools).map((tool) => (
              <button
                key={tool.name}
                type="button"
                className={styles.dropdownItem}
                onClick={() => {
                  setIsToolsOpen(false);
                  if (tool.type === 'url' && tool.url) {
                    window.open(tool.url, '_blank', 'noopener,noreferrer');
                  }
                }}
              >
                <span className={styles.dropdownIcon}>
                  {tool.iconPath ? <img src={tool.iconPath} alt={tool.name} /> : tool.type === 'action' ? 'R' : 'OK'}
                </span>
                <span className={styles.dropdownText}>{tool.name}</span>
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <button
        type="button"
        className={styles.featureFlagsBtn}
        onClick={onOpenFeatureFlags}
        data-testid="feature-flags-button"
      >
        Feature Flags
      </button>
    </>
  );
}
