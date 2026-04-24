import { useReducer, useRef, useEffect, useCallback, type ReactElement } from 'react';
import type {
  GenericFilterDefinition,
  GenericFilterParameter,
  GenericFilterParameterOption,
} from '@/gen/image_processor_service_pb';
import { useFilters } from '@/presentation/hooks/useFilters';
import { useToast } from '@/presentation/hooks/useToast';
import styles from './FilterPanel.module.css';

export interface ActiveFilterState {
  id: string;
  parameters: Record<string, string>;
}

interface FilterPanelProps {
  filters?: GenericFilterDefinition[];
  onFiltersChange: (activeFilters: ActiveFilterState[]) => void;
  initialActiveFilters?: ActiveFilterState[];
  /** When incremented, refetches filter definitions from the backend (processor service updates). */
  processorFilterEpoch?: number;
  disabled?: boolean;
}

interface FilterState {
  id: string;
  name: string;
  enabled: boolean;
  expanded: boolean;
  parameters: GenericFilterParameter[];
  parameterValues: Record<string, string>;
}

// --- Reducer ---

interface FilterPanelState {
  filters: FilterState[];
  draggedIndex: number | null;
  dragOverIndex: number | null;
}

enum FilterActionType {
  INIT = 'INIT',
  TOGGLE_CARD = 'TOGGLE_CARD',
  SET_ENABLED = 'SET_ENABLED',
  SET_PARAMETER = 'SET_PARAMETER',
  REORDER = 'REORDER',
  DRAG_START = 'DRAG_START',
  DRAG_ENTER = 'DRAG_ENTER',
  DRAG_END = 'DRAG_END',
}

type FilterAction =
  | { type: FilterActionType.INIT; payload: FilterState[] }
  | { type: FilterActionType.TOGGLE_CARD; index: number }
  | { type: FilterActionType.SET_ENABLED; index: number; enabled: boolean }
  | { type: FilterActionType.SET_PARAMETER; filterId: string; paramId: string; value: string }
  | { type: FilterActionType.REORDER; from: number; to: number }
  | { type: FilterActionType.DRAG_START; index: number }
  | { type: FilterActionType.DRAG_ENTER; index: number }
  | { type: FilterActionType.DRAG_END };

function filterPanelReducer(state: FilterPanelState, action: FilterAction): FilterPanelState {
  switch (action.type) {
    case FilterActionType.INIT:
      return { filters: action.payload, draggedIndex: null, dragOverIndex: null };

    case FilterActionType.TOGGLE_CARD: {
      const filters = [...state.filters];
      const filter = filters[action.index];
      if (!filter) return state;
      const expanding = !filter.expanded;
      filters[action.index] = { ...filter, expanded: expanding, enabled: expanding ? true : filter.enabled };
      return { ...state, filters };
    }

    case FilterActionType.SET_ENABLED: {
      const filters = [...state.filters];
      const filter = filters[action.index];
      if (!filter) return state;
      filters[action.index] = { ...filter, enabled: action.enabled, expanded: action.enabled ? filter.expanded : false };
      return { ...state, filters };
    }

    case FilterActionType.SET_PARAMETER: {
      const filters = state.filters.map((f) =>
        f.id === action.filterId
          ? { ...f, parameterValues: { ...f.parameterValues, [action.paramId]: action.value } }
          : f
      );
      return { ...state, filters };
    }

    case FilterActionType.REORDER: {
      const filters = [...state.filters];
      const [moved] = filters.splice(action.from, 1);
      filters.splice(action.to, 0, moved);
      return { ...state, filters, draggedIndex: null, dragOverIndex: null };
    }

    case FilterActionType.DRAG_START:
      return { ...state, draggedIndex: action.index };

    case FilterActionType.DRAG_ENTER:
      return { ...state, dragOverIndex: action.index };

    case FilterActionType.DRAG_END:
      return { ...state, draggedIndex: null, dragOverIndex: null };

    default:
      return state;
  }
}

// --- Helpers ---

function parseParamMetadata(
  metadata: Record<string, string>,
  defaults: { min: number; max: number; step: number }
) {
  return {
    min: metadata.min !== undefined ? parseFloat(metadata.min) : defaults.min,
    max: metadata.max !== undefined ? parseFloat(metadata.max) : defaults.max,
    step: metadata.step !== undefined ? parseFloat(metadata.step) : defaults.step,
  };
}

function getActiveFilters(filterStates: FilterState[]): ActiveFilterState[] {
  const active = filterStates.filter((f) => f.enabled);
  if (active.length === 0) {
    return [{ id: 'none', parameters: {} }];
  }
  return active.map((filter) => ({ id: filter.id, parameters: { ...filter.parameterValues } }));
}

// --- Parameter sub-components ---

interface ParameterProps {
  filter: FilterState;
  param: GenericFilterParameter;
  onChange: (filterId: string, paramId: string, value: string) => void;
}

interface NumberParameterProps extends ParameterProps {
  onError: (title: string, message: string) => void;
}

function SelectParameter({ filter, param, onChange }: ParameterProps): ReactElement {
  const currentValue = filter.parameterValues[param.id] || param.defaultValue || '';
  return (
    <div className={styles.paramControl}>
      <label className={styles.paramLabel}>{param.name}</label>
      <div className={styles.radioGroup}>
        {param.options?.map((option: GenericFilterParameterOption) => (
          <label key={option.value} className={styles.radioOption}>
            <input
              type="radio"
              name={`${filter.id}-${param.id}`}
              value={option.value}
              checked={currentValue === option.value}
              onChange={() => onChange(filter.id, param.id, option.value)}
              draggable={false}
              data-testid={`filter-parameter-${filter.id}-${param.id}-${option.value}`}
            />
            <span>{option.label || option.value}</span>
          </label>
        ))}
      </div>
    </div>
  );
}

function RangeParameter({ filter, param, onChange }: ParameterProps): ReactElement {
  const currentValue = filter.parameterValues[param.id] || param.defaultValue || '';
  const { min, max, step } = parseParamMetadata(param.metadata || {}, { min: 3, max: 15, step: 2 });
  const rangeValue = currentValue ? parseFloat(currentValue) : (param.defaultValue ? parseFloat(param.defaultValue) : min);
  return (
    <div className={styles.paramControl}>
      <label className={styles.paramLabel}>{param.name}</label>
      <div className={styles.sliderContainer}>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={rangeValue}
          onChange={(e) => onChange(filter.id, param.id, e.target.value)}
          draggable={false}
          data-testid={`filter-parameter-${filter.id}-${param.id}`}
        />
        <span className={styles.sliderValue}>{rangeValue}</span>
      </div>
    </div>
  );
}

function NumberParameter({ filter, param, onChange, onError }: NumberParameterProps): ReactElement {
  const currentValue = filter.parameterValues[param.id] || param.defaultValue || '';
  const { min, max, step } = parseParamMetadata(param.metadata || {}, { min: -Infinity, max: Infinity, step: 1 });
  const numValue = currentValue || param.defaultValue || (min !== -Infinity ? String(min) : '0');

  function handleChange(inputValue: string) {
    if (inputValue === '' || inputValue === '-') return;
    const parsed = parseFloat(inputValue);
    if (isNaN(parsed)) return;
    let clamped = parsed;
    if (min !== -Infinity && parsed < min) {
      clamped = min;
      onError('Invalid Value', `${param.name} must be at least ${min}. Value adjusted to ${min}.`);
    } else if (max !== Infinity && parsed > max) {
      clamped = max;
      onError('Invalid Value', `${param.name} must be at most ${max}. Value adjusted to ${max}.`);
    }
    onChange(filter.id, param.id, String(clamped));
  }

  return (
    <div className={styles.paramControl}>
      <label className={styles.paramLabel}>{param.name}</label>
      <div className={styles.numberInputContainer}>
        <input
          type="number"
          min={min !== -Infinity ? min : undefined}
          max={max !== Infinity ? max : undefined}
          step={step}
          value={numValue}
          onChange={(e) => handleChange(e.target.value)}
          draggable={false}
          onKeyDown={(e) => {
            if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
              e.preventDefault();
              const current = parseFloat(e.currentTarget.value) || 0;
              const next = e.key === 'ArrowUp' ? current + step : current - step;
              if ((min !== -Infinity && next < min) || (max !== Infinity && next > max)) return;
              onChange(filter.id, param.id, String(next));
            }
          }}
          onWheel={(e) => {
            e.preventDefault();
            const current = parseFloat(e.currentTarget.value) || 0;
            const next = Math.max(min, Math.min(max, current + (e.deltaY < 0 ? step : -step)));
            onChange(filter.id, param.id, String(next));
          }}
          data-testid={`filter-parameter-${filter.id}-${param.id}`}
        />
      </div>
    </div>
  );
}

function CheckboxParameter({ filter, param, onChange }: ParameterProps): ReactElement {
  const currentValue = filter.parameterValues[param.id] || '';
  const checked = currentValue === 'true' || (param.defaultValue === 'true' && !currentValue);
  return (
    <div className={styles.paramControl}>
      <label className={styles.checkboxOption}>
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(filter.id, param.id, e.target.checked.toString())}
          draggable={false}
          data-testid={`filter-parameter-${filter.id}-${param.id}`}
        />
        <span>{param.name}</span>
      </label>
    </div>
  );
}

function renderParameterControl(
  filter: FilterState,
  param: GenericFilterParameter,
  onChange: (filterId: string, paramId: string, value: string) => void,
  onError: (title: string, message: string) => void
): ReactElement {
  switch (param.type) {
    case 1: return <SelectParameter key={param.id} filter={filter} param={param} onChange={onChange} />;
    case 2: return <RangeParameter key={param.id} filter={filter} param={param} onChange={onChange} />;
    case 3: return <NumberParameter key={param.id} filter={filter} param={param} onChange={onChange} onError={onError} />;
    case 4: return <CheckboxParameter key={param.id} filter={filter} param={param} onChange={onChange} />;
    default: return <></>;
  }
}

// --- Main component ---

export function FilterPanel({
  filters: propFilters,
  onFiltersChange,
  initialActiveFilters,
  processorFilterEpoch,
  disabled,
}: FilterPanelProps): ReactElement {
  const { filters: availableFilters, refetch } = useFilters();
  const filters = propFilters ?? availableFilters;

  useEffect(() => {
    if (processorFilterEpoch === undefined || processorFilterEpoch < 1) return;
    void refetch();
  }, [processorFilterEpoch, refetch]);

  const [{ filters: localFilters, draggedIndex, dragOverIndex }, dispatch] = useReducer(
    filterPanelReducer,
    { filters: [], draggedIndex: null, dragOverIndex: null }
  );

  // draggedIndexRef allows handleDrop to read the current index synchronously
  // without a stale closure, since useReducer state updates are async.
  const draggedIndexRef = useRef<number | null>(null);
  const toastTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  // Track filter definition IDs — only re-initialize when backend sends new filter definitions,
  // not on every re-render triggered by activeFilters updates (avoids circular reset).
  const filterIdsKeyRef = useRef<string>('');
  const initialActiveFiltersRef = useRef(initialActiveFilters);

  useEffect(() => {
    const newKey = filters.map((f) => f.id).join(',');
    if (filterIdsKeyRef.current === newKey) return;
    filterIdsKeyRef.current = newKey;

    const initialFilterStates: FilterState[] = filters.map((filter) => ({
      id: filter.id,
      name: filter.name,
      enabled: false,
      expanded: false,
      parameters: filter.parameters,
      parameterValues: filter.parameters.reduce((acc, param) => {
        acc[param.id] = param.defaultValue || '';
        return acc;
      }, {} as Record<string, string>),
    }));

    const initActive = initialActiveFiltersRef.current;
    if (initActive && initActive.length > 0) {
      initActive.forEach((active) => {
        const filter = initialFilterStates.find((f) => f.id === active.id);
        if (filter) {
          filter.enabled = true;
          filter.expanded = true;
          filter.parameterValues = { ...filter.parameterValues, ...active.parameters };
        }
      });
    }

    dispatch({ type: FilterActionType.INIT, payload: initialFilterStates });
  }, [filters]);

  const { error: showError } = useToast();

  const debouncedError = useCallback(
    (title: string, message: string) => {
      if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
      toastTimeoutRef.current = setTimeout(() => showError(title, message), 100);
    },
    [showError]
  );

  const handleParameterChange = useCallback((filterId: string, paramId: string, value: string) => {
    dispatch({ type: FilterActionType.SET_PARAMETER, filterId, paramId, value });
  }, []);

  useEffect(() => {
    if (localFilters.length === 0) return;
    onFiltersChange(getActiveFilters(localFilters));
  }, [localFilters, onFiltersChange]);

  const handleDragStart = useCallback((e: React.DragEvent, index: number) => {
    draggedIndexRef.current = index;
    dispatch({ type: FilterActionType.DRAG_START, index });
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', String(index));
    e.dataTransfer.setDragImage(e.currentTarget as HTMLElement, 20, 20);
  }, []);

  const handleDragEnd = useCallback(() => {
    draggedIndexRef.current = null;
    dispatch({ type: FilterActionType.DRAG_END });
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent, index: number) => {
    e.preventDefault();
    dispatch({ type: FilterActionType.DRAG_ENTER, index });
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      dispatch({ type: FilterActionType.DRAG_END });
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, to: number) => {
    e.preventDefault();
    const from = draggedIndexRef.current;
    draggedIndexRef.current = null;
    if (from === null || from === to) {
      dispatch({ type: FilterActionType.DRAG_END });
      return;
    }
    dispatch({ type: FilterActionType.REORDER, from, to });
  }, []);

  if (localFilters.length === 0) {
    return <div className={styles.noFilters}>No filters available</div>;
  }

  return (
    <div className={`${styles.controlSection}${disabled ? ` ${styles.disabled}` : ''}`}>
      <label className={styles.controlLabel}>
        Filters <span className={styles.hint}>(drag to reorder)</span>
      </label>
      <div className={styles.filtersList}>
        {localFilters.map((filter, index) => (
          <div
            key={filter.id}
            draggable={true}
            className={[
              styles.filterCard,
              draggedIndex === index ? styles.dragging : '',
              dragOverIndex === index && draggedIndex !== index ? styles.dragOver : '',
            ].join(' ')}
            onDragStart={(e) => handleDragStart(e, index)}
            onDragEnd={handleDragEnd}
            onDragOver={handleDragOver}
            onDragEnter={(e) => handleDragEnter(e, index)}
            onDragLeave={handleDragLeave}
            onDrop={(e) => handleDrop(e, index)}
            data-filter-name={filter.name}
            data-filter-id={filter.id}
          >
            <div className={styles.filterHeader} onClick={() => dispatch({ type: FilterActionType.TOGGLE_CARD, index })}>
              <div className={styles.dragHandleContainer} onClick={(e) => e.stopPropagation()}>
                <span className={styles.dragHandle}>⋮⋮</span>
              </div>
              <div className={styles.checkboxContainer} onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={filter.enabled}
                  onChange={(e) => dispatch({ type: FilterActionType.SET_ENABLED, index, enabled: e.target.checked })}
                  onClick={(e) => e.stopPropagation()}
                  draggable={false}
                  data-testid={`filter-checkbox-${filter.id}`}
                />
              </div>
              <label>{filter.name}</label>
              <span className={`${styles.chevron} ${filter.expanded ? styles.expanded : ''}`}>▶</span>
            </div>

            {filter.parameters.length > 0 && (
              <div className={`${styles.filterBody} ${filter.expanded ? styles.expanded : ''}`}>
                {filter.parameters.map((param) =>
                  renderParameterControl(filter, param, handleParameterChange, debouncedError)
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
