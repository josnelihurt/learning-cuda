import { FilterPanel } from '../filters/FilterPanel';
import { useDashboardState } from '../../context/dashboard-state-context';
import { SidebarControls } from './SidebarControls';

export function SidebarColumn() {
  const { setActiveFilters, processorFilterEpoch, activeFilters } = useDashboardState();

  return (
    <aside className="sidebar">
      <div className="sidebar-content">
        <SidebarControls />
        <div data-testid="react-filters-section">
          <FilterPanel
            processorFilterEpoch={processorFilterEpoch}
            onFiltersChange={setActiveFilters}
            initialActiveFilters={activeFilters}
          />
        </div>
      </div>
    </aside>
  );
}
