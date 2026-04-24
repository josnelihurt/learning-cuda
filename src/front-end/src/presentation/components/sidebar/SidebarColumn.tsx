import { useEffect, useReducer, type ReactElement } from 'react';
import { FilterPanel } from '@/presentation/components/filters/FilterPanel';
import { useDashboardState } from '@/presentation/context/dashboard-state-context';
import { SidebarControls } from './SidebarControls';
import styles from './SidebarColumn.module.css';

type SidebarColumnState = { expanded: boolean };

enum SidebarActionType {
  EXPAND = 'EXPAND',
  COLLAPSE = 'COLLAPSE',
}

type SidebarColumnAction = { type: SidebarActionType.EXPAND } | { type: SidebarActionType.COLLAPSE };

const INITIAL_SIDEBAR_STATE: SidebarColumnState = { expanded: true };

function sidebarColumnReducer(state: SidebarColumnState, action: SidebarColumnAction): SidebarColumnState {
  switch (action.type) {
    case SidebarActionType.EXPAND:
      return { expanded: true };
    case SidebarActionType.COLLAPSE:
      return { expanded: false };
    default:
      return state;
  }
}

export function SidebarColumn(): ReactElement {
  const { setActiveFilters, processorFilterEpoch, activeFilters, selectedSourceNumber, isWebRTCReady } = useDashboardState();
  const [state, dispatch] = useReducer(sidebarColumnReducer, INITIAL_SIDEBAR_STATE);

  useEffect(() => {
    document.body.classList.toggle('sidebar-column-collapsed', !state.expanded);
    return () => {
      document.body.classList.remove('sidebar-column-collapsed');
    };
  }, [state.expanded]);

  const asideClassName = state.expanded
    ? `sidebar ${styles.aside}`
    : `sidebar ${styles.aside} ${styles.asideCollapsed}`;

  const contentClassName = state.expanded
    ? 'sidebar-content'
    : `sidebar-content ${styles.contentHidden}`;

  return (
    <>
      {state.expanded ? (
        <button
          type="button"
          className={styles.grip}
          data-testid="sidebar-column-collapse"
          aria-label="Collapse sidebar"
          aria-expanded
          onClick={() => dispatch({ type: SidebarActionType.COLLAPSE })}
        >
          <span className={styles.chevronLeft} aria-hidden />
        </button>
      ) : (
        <button
          type="button"
          className={styles.grip}
          data-testid="sidebar-column-expand"
          aria-label="Expand sidebar"
          aria-expanded={false}
          onClick={() => dispatch({ type: SidebarActionType.EXPAND })}
        >
          <span className={styles.chevronRight} aria-hidden />
        </button>
      )}
      <aside className={asideClassName}>
        <div className={contentClassName}>
          <SidebarControls />
          <div data-testid="react-filters-section">
            <FilterPanel
              key={selectedSourceNumber}
              processorFilterEpoch={processorFilterEpoch}
              onFiltersChange={setActiveFilters}
              initialActiveFilters={activeFilters}
              disabled={!isWebRTCReady}
            />
          </div>
        </div>
      </aside>
    </>
  );
}
