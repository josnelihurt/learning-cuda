import type { FilterDefinition } from '../../gen/common_pb';
import type { Filter } from '../../components/app/filter-panel.types';

export interface IProcessorCapabilitiesService {
  initialize(): Promise<void>;
  getFilters(): Filter[];
  getFilterDefinitions(): FilterDefinition[];
  isInitialized(): boolean;
}
