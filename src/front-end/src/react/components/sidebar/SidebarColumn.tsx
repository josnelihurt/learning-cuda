import { FilterPanel } from '../filters/FilterPanel';
import { useDashboardState } from '../../context/dashboard-state-context';
import { SidebarControls } from './SidebarControls';

export function SidebarColumn() {
  const { setActiveFilters, processorFilterEpoch, activeFilters } = useDashboardState();

  return (
    <aside className="sidebar">
      <div className="sidebar-content">
        <SidebarControls />
        <FilterPanel
          processorFilterEpoch={processorFilterEpoch}
          onFiltersChange={setActiveFilters}
          initialActiveFilters={activeFilters}
        />
      </div>
    </aside>
  );
}
