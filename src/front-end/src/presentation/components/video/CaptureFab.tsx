import { type ReactElement } from 'react';
import styles from './CaptureFab.module.css';

type CaptureFabProps = {
  onClick: () => void;
};

export function CaptureFab({ onClick }: CaptureFabProps): ReactElement {
  return (
    <button
      type="button"
      className={styles.fab}
      onClick={onClick}
      data-testid="capture-fab"
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
        <circle cx="12" cy="13" r="4" />
      </svg>
      <span>Capture</span>
    </button>
  );
}
