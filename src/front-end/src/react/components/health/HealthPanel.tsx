import { useState } from 'react';
import { memo } from 'react';
import styles from './HealthPanel.module.css';
import type { GrpcAsyncError } from '../../hooks/useAsyncGRPC';

export interface HealthPanelProps {
  isHealthy: boolean;
  loading: boolean;
  error: GrpcAsyncError | null;
  lastChecked: Date | undefined;
  message?: string;
  compact?: boolean;
}

/**
 * HealthPanel displays detailed health information with status, timestamp, and error details.
 *
 * @param isHealthy - Health status from useHealthMonitor
 * @param loading - Loading state
 * @param error - Error message from gRPC call
 * @param lastChecked - Last check timestamp
 * @param message - Status message from response
 * @param compact - Compact mode (optional, default: false)
 */
export const HealthPanel = memo(function HealthPanel({
  isHealthy,
  loading,
  error,
  lastChecked,
  message,
  compact = false,
}: HealthPanelProps) {
  const [expanded, setExpanded] = useState(false);

  const getStatusIcon = () => {
    if (loading) return '⏳';
    return isHealthy ? '✓' : '⚠';
  };

  const getStatusText = () => {
    if (loading) return 'Checking...';
    return isHealthy ? 'Healthy' : 'Unhealthy';
  };

  const formatLastChecked = (date?: Date): string => {
    if (!date) return 'No checks yet';
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} min ago`;
    if (date.toDateString() === now.toDateString()) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const toggleExpanded = () => {
    if (compact) {
      setExpanded(!expanded);
    }
  };

  if (compact) {
    return (
      <div
        className={`${styles.panel} ${styles.compact} ${expanded ? styles.expanded : ''} ${isHealthy ? styles.healthy : styles.unhealthy}`}
        onClick={toggleExpanded}
        role="button"
        tabIndex={0}
        onKeyPress={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleExpanded();
          }
        }}
        aria-expanded={expanded}
      >
        <div className={styles.compactRow}>
          <span className={styles.icon} aria-hidden="true">
            {getStatusIcon()}
          </span>
          <span className={styles.status}>{getStatusText()}</span>
          <span className={styles.timestamp}>{formatLastChecked(lastChecked)}</span>
        </div>
        {expanded && (
          <div className={styles.details}>
            {message && <div className={styles.message}>{message}</div>}
            {error && (
              <div className={styles.error}>
                <div className={styles.errorTitle}>Error:</div>
                <div className={styles.errorMessage}>{error.message}</div>
              </div>
            )}
            <div className={styles.timestampDetail}>Last checked: {lastChecked?.toLocaleString() || 'Never'}</div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`${styles.panel} ${isHealthy ? styles.healthy : styles.unhealthy}`}>
      <div className={styles.header}>
        <h3 className={styles.title}>Backend Health</h3>
      </div>

      <div className={styles.statusRow}>
        <span className={`${styles.icon} ${styles.largeIcon}`} aria-hidden="true">
          {getStatusIcon()}
        </span>
        <span className={styles.statusText}>{getStatusText()}</span>
      </div>

      {!loading && (
        <div className={styles.details}>
          {message && (
            <div className={styles.message}>
              <span className={styles.messageLabel}>Status:</span>
              <span className={styles.messageValue}>{message}</span>
            </div>
          )}

          {error && (
            <div className={styles.error}>
              <div className={styles.errorTitle}>Error Details:</div>
              <div className={styles.errorMessage}>{error.message}</div>
              {error.code && <div className={styles.errorCode}>Code: {error.code}</div>}
            </div>
          )}

          <div className={styles.timestamp}>
            Last checked: {lastChecked?.toLocaleString() || 'Never'}
          </div>
        </div>
      )}
    </div>
  );
});
