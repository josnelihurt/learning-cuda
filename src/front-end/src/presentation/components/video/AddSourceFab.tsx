import { type ReactElement } from 'react';
import styles from './AddSourceFab.module.css';

type AddSourceFabProps = {
  onClick: () => void;
};

export function AddSourceFab({ onClick }: AddSourceFabProps): ReactElement {
  return (
    <button
      type="button"
      className={styles.fab}
      onClick={onClick}
      data-testid="add-input-fab"
    >
      <span>+</span>
      <span>Add Input</span>
    </button>
  );
}
