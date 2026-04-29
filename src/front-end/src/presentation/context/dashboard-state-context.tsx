import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import { useAppServices } from '@/presentation/providers/app-services-provider';
import { AcceleratorType } from '@/gen/common_pb';

export type DashboardState = {
  selectedSourceNumber: number;
  selectedSourceName: string;
  selectedAccelerator: AcceleratorType;
  selectedResolution: string;
  activeFilters: ActiveFilterState[];
  processorFilterEpoch: number;
  isWebRTCReady: boolean;
  setSelectedSource: (number: number, name: string) => void;
  setAccelerator: (a: AcceleratorType) => void;
  setResolution: (r: string) => void;
  setActiveFilters: (f: ActiveFilterState[]) => void;
  setWebRTCReady: (ready: boolean) => void;
};

const DashboardStateContext = createContext<DashboardState | null>(null);

export function useDashboardState(): DashboardState {
  const ctx = useContext(DashboardStateContext);
  if (!ctx) {
    throw new Error('useDashboardState must be used within DashboardStateProvider');
  }
  return ctx;
}

export function DashboardStateProvider({ children }: { children: ReactNode }): ReactNode {
  const { container, ready } = useAppServices();
  const [selectedSourceNumber, setSelectedSourceNumber] = useState(1);
  const [selectedSourceName, setSelectedSourceName] = useState('Lena');
  const [selectedAccelerator, setSelectedAccelerator] = useState<AcceleratorType>(AcceleratorType.CUDA);
  const [selectedResolution, setSelectedResolution] = useState('original');
  const [activeFilters, setActiveFiltersState] = useState<ActiveFilterState[]>([]);
  const [processorFilterEpoch, setProcessorFilterEpoch] = useState(0);
  const [isWebRTCReady, setIsWebRTCReady] = useState(false);

  useEffect(() => {
    if (!ready) {
      return;
    }
    const svc = container.getProcessorCapabilitiesService();
    const onFiltersUpdated = (): void => {
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

  const setAccelerator = useCallback((a: AcceleratorType) => {
    setSelectedAccelerator(a);
  }, []);

  const setResolution = useCallback((r: string) => {
    setSelectedResolution(r);
  }, []);

  const setActiveFilters = useCallback((f: ActiveFilterState[]) => {
    setActiveFiltersState(f);
  }, []);

  const setWebRTCReady = useCallback((ready: boolean) => {
    setIsWebRTCReady(ready);
  }, []);

  const value = useMemo(
    () => ({
      selectedSourceNumber,
      selectedSourceName,
      selectedAccelerator,
      selectedResolution,
      activeFilters,
      processorFilterEpoch,
      isWebRTCReady,
      setSelectedSource,
      setAccelerator,
      setResolution,
      setActiveFilters,
      setWebRTCReady,
    }),
    [
      selectedSourceNumber,
      selectedSourceName,
      selectedAccelerator,
      selectedResolution,
      activeFilters,
      processorFilterEpoch,
      isWebRTCReady,
      setSelectedSource,
      setAccelerator,
      setResolution,
      setActiveFilters,
      setWebRTCReady,
    ]
  );

  return (
    <DashboardStateContext.Provider value={value}>{children}</DashboardStateContext.Provider>
  );
}
