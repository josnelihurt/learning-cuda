import { useDashboardState } from '../../context/dashboard-state-context';
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
        <span className={styles.controlLabel}>Accelerator</span>
        <div className={styles.segmented}>
          <button
            type="button"
            className={`${styles.segment} ${selectedAccelerator === 'gpu' ? styles.segmentActive : ''}`}
            data-value="gpu"
            onClick={() => setAccelerator('gpu')}
          >
            GPU
          </button>
          <button
            type="button"
            className={`${styles.segment} ${selectedAccelerator === 'cpu' ? styles.segmentActive : ''}`}
            data-value="cpu"
            onClick={() => setAccelerator('cpu')}
          >
            CPU
          </button>
        </div>
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
