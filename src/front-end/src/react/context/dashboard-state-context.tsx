import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import type { ActiveFilterState } from '../components/filters/FilterPanel';
import { useAppServices } from '../providers/app-services-provider';

export type AcceleratorChoice = 'gpu' | 'cpu';

export type DashboardState = {
  selectedSourceNumber: number;
  selectedSourceName: string;
  selectedAccelerator: AcceleratorChoice;
  selectedResolution: string;
  activeFilters: ActiveFilterState[];
  processorFilterEpoch: number;
  setSelectedSource: (number: number, name: string) => void;
  setAccelerator: (a: AcceleratorChoice) => void;
  setResolution: (r: string) => void;
  setActiveFilters: (f: ActiveFilterState[]) => void;
};

const DashboardStateContext = createContext<DashboardState | null>(null);

export function useDashboardState(): DashboardState {
  const ctx = useContext(DashboardStateContext);
  if (!ctx) {
    throw new Error('useDashboardState must be used within DashboardStateProvider');
  }
  return ctx;
}

export function DashboardStateProvider({ children }: { children: ReactNode }) {
  const { container, ready } = useAppServices();
  const [selectedSourceNumber, setSelectedSourceNumber] = useState(1);
  const [selectedSourceName, setSelectedSourceName] = useState('Lena');
  const [selectedAccelerator, setSelectedAccelerator] = useState<AcceleratorChoice>('gpu');
  const [selectedResolution, setSelectedResolution] = useState('original');
  const [activeFilters, setActiveFiltersState] = useState<ActiveFilterState[]>([]);
  const [processorFilterEpoch, setProcessorFilterEpoch] = useState(0);

  useEffect(() => {
    if (!ready) {
      return;
    }
    const svc = container.getProcessorCapabilitiesService();
    const onFiltersUpdated = () => {
      setProcessorFilterEpoch((n) => n + 1);
    };
    svc.addFiltersUpdatedListener(onFiltersUpdated);
    return () => {
      svc.removeFiltersUpdatedListener(onFiltersUpdated);
    };
  }, [container, ready]);

  const setSelectedSource = useCallback((n: number, name: string) => {
    setSelectedSourceNumber(n);
    setSelectedSourceName(name);
  }, []);

  const setAccelerator = useCallback((a: AcceleratorChoice) => {
    setSelectedAccelerator(a);
  }, []);

  const setResolution = useCallback((r: string) => {
    setSelectedResolution(r);
  }, []);

  const setActiveFilters = useCallback((f: ActiveFilterState[]) => {
    setActiveFiltersState(f);
  }, []);

  const value = useMemo(
    () => ({
      selectedSourceNumber,
      selectedSourceName,
      selectedAccelerator,
      selectedResolution,
      activeFilters,
      processorFilterEpoch,
      setSelectedSource,
      setAccelerator,
      setResolution,
      setActiveFilters,
    }),
    [
      selectedSourceNumber,
      selectedSourceName,
      selectedAccelerator,
      selectedResolution,
      activeFilters,
      processorFilterEpoch,
      setSelectedSource,
      setAccelerator,
      setResolution,
      setActiveFilters,
    ]
  );

  return (
    <DashboardStateContext.Provider value={value}>{children}</DashboardStateContext.Provider>
  );
}
