import { memo } from 'react';
import styles from './HealthIndicator.module.css';

export interface HealthIndicatorProps {
  isHealthy: boolean;
  loading?: boolean;
  onClick?: () => void;
  showLabel?: boolean;
  lastChecked?: Date;
  message?: string;
}

/**
 * HealthIndicator displays a compact status indicator with a colored dot and optional label.
 *
 * @param isHealthy - Health status from useHealthMonitor
 * @param loading - Loading state (optional, default: false)
 * @param onClick - Click handler (optional)
 * @param showLabel - Whether to show text label (optional, default: true)
 * @param lastChecked - Last check timestamp for tooltip (optional)
 * @param message - Status message for tooltip (optional)
 */
export const HealthIndicator = memo(function HealthIndicator({
  isHealthy,
  loading = false,
  onClick,
  showLabel = true,
  lastChecked,
  message,
}: HealthIndicatorProps) {
  const getStatusClass = () => {
    if (loading) return 'loading';
    return isHealthy ? 'healthy' : 'unhealthy';
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

    if (diffMins < 1) return 'Last checked: Just now';
    if (diffMins < 60) return `Last checked: ${diffMins} min ago`;
    if (date.toDateString() === now.toDateString()) {
      return `Last checked: ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    return `Last checked: ${date.toLocaleString()}`;
  };

  const tooltipContent = message || formatLastChecked(lastChecked);

  return (
    <div
      className={`${styles.indicator} ${styles[getStatusClass()]} ${onClick ? styles.clickable : ''}`}
      onClick={onClick}
      title={tooltipContent}
      role="status"
      aria-label={getStatusText()}
      aria-live="polite"
    >
      <div className={styles.dot} aria-hidden="true" />
      {showLabel && <span className={styles.label}>{getStatusText()}</span>}
    </div>
  );
});
