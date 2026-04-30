import { useEffect } from 'react';
import { useDashboardState } from '@/presentation/context/dashboard-state-context';
import { useAcceleratorCapabilities } from '@/presentation/hooks/useAcceleratorCapabilities';
import { AcceleratorType } from '@/gen/common_pb';
import styles from './SidebarControls.module.css';

export function SidebarControls() {
  const {
    selectedSourceNumber,
    selectedSourceName,
    selectedAccelerator,
    selectedResolution,
    setAccelerator,
    setResolution,
  } = useDashboardState();
  const { options, loading } = useAcceleratorCapabilities();

  // Auto-select the first available accelerator when capabilities load and the
  // current selection isn't in the list (e.g. default CUDA with a Vulkan build).
  useEffect(() => {
    if (loading || options.length === 0) return;
    const isCurrentValid = options.some((o) => o.value === selectedAccelerator);
    if (!isCurrentValid) {
      setAccelerator(options[0].value);
    }
  }, [loading, options, selectedAccelerator, setAccelerator]);

  return (
    <div className={styles.shell}>
      <div className={styles.controlSection}>
        <span className={styles.controlLabel}>Selected</span>
        <div className={styles.selectedRow}>
          <span className={styles.sourceBadge}>{selectedSourceNumber}</span>
          <span>{selectedSourceName}</span>
        </div>
      </div>

      <div className={styles.controlSection}>
        <label className={styles.controlLabel} htmlFor="accelerator-select-react">
          Accelerator
        </label>
        <select
          id="accelerator-select-react"
          className={styles.compactSelect}
          data-testid="accelerator-select"
          value={selectedAccelerator}
          onChange={(e) => setAccelerator(Number(e.target.value) as AcceleratorType)}
          disabled={loading || options.length === 0}
        >
          {loading && <option value="">Loading...</option>}
          {!loading && options.length === 0 && <option value="">No accelerator connected</option>}
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.controlSection}>
        <label className={styles.controlLabel} htmlFor="resolution-select-react">
          Resolution
        </label>
        <select
          id="resolution-select-react"
          className={styles.compactSelect}
          data-testid="resolution-select"
          value={selectedResolution}
          onChange={(e) => setResolution(e.target.value)}
        >
          <option value="original">Original Size</option>
          <option value="half">Half</option>
          <option value="quarter">Quarter</option>
        </select>
      </div>
    </div>
  );
}
