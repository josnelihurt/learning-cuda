import { useState, useRef, useEffect, useCallback } from 'react';
import type {
  GenericFilterDefinition,
  GenericFilterParameter,
  GenericFilterParameterOption,
} from '@/gen/image_processor_service_pb';
import { useFilters } from '../../hooks/useFilters';
import { useToast } from '../../hooks/useToast';
import styles from './FilterPanel.module.css';

export interface ActiveFilterState {
  id: string;
  parameters: Record<string, string>;
}

interface FilterPanelProps {
  filters?: GenericFilterDefinition[];
  onFiltersChange: (activeFilters: ActiveFilterState[]) => void;
  initialActiveFilters?: ActiveFilterState[];
}

interface FilterState {
  id: string;
  name: string;
  enabled: boolean;
  expanded: boolean;
  parameters: GenericFilterParameter[];
  parameterValues: Record<string, string>;
}

export function FilterPanel({
  filters: propFilters,
  onFiltersChange,
  initialActiveFilters,
}: FilterPanelProps) {
  const { filters: availableFilters } = useFilters();
  const filters = propFilters || availableFilters;

  const [localFilters, setLocalFilters] = useState<FilterState[]>([]);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const toastTimeoutRef = useRef<NodeJS.Timeout>();

  // Initialize local filters from prop filters
  useEffect(() => {
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

    // Apply initial active filters if provided
    if (initialActiveFilters && initialActiveFilters.length > 0) {
      initialActiveFilters.forEach((active) => {
        const filter = initialFilterStates.find((f) => f.id === active.id);
        if (filter) {
          filter.enabled = true;
          filter.expanded = true;
          filter.parameterValues = { ...filter.parameterValues, ...active.parameters };
        }
      });
    }

    setLocalFilters(initialFilterStates);
  }, [filters, initialActiveFilters]);

  const { error: showError } = useToast();

  const debouncedError = useCallback(
    (title: string, message: string) => {
      if (toastTimeoutRef.current) {
        clearTimeout(toastTimeoutRef.current);
      }
      toastTimeoutRef.current = setTimeout(() => {
        showError(title, message);
      }, 100);
    },
    [showError]
  );

  const toggleCard = useCallback(
    (index: number) => {
      setLocalFilters((prev) => {
        const newFilters = [...prev];
        const filter = { ...newFilters[index] };

        if (!filter.expanded) {
          // Expand - auto-enable
          filter.expanded = true;
          filter.enabled = true;
        } else {
          // Collapse
          filter.expanded = false;
        }

        newFilters[index] = filter;
        onFiltersChange(getActiveFilters(newFilters));
        return newFilters;
      });
    },
    [onFiltersChange]
  );

  const handleCheckboxChange = useCallback(
    (index: number, enabled: boolean) => {
      setLocalFilters((prev) => {
        const newFilters = [...prev];
        newFilters[index] = {
          ...newFilters[index],
          enabled,
          expanded: enabled ? newFilters[index].expanded : false,
        };
        onFiltersChange(getActiveFilters(newFilters));
        return newFilters;
      });
    },
    [onFiltersChange]
  );

  const handleParameterChange = useCallback(
    (filterId: string, paramId: string, value: string) => {
      setLocalFilters((prev) => {
        const newFilters = prev.map((filter) => {
          if (filter.id === filterId) {
            return {
              ...filter,
              parameterValues: {
                ...filter.parameterValues,
                [paramId]: value,
              },
            };
          }
          return filter;
        });
        onFiltersChange(getActiveFilters(newFilters));
        return newFilters;
      });
    },
    [onFiltersChange]
  );

  const handleNumberInputChange = useCallback(
    (filterId: string,
     paramId: string,
     param: GenericFilterParameter,
     inputValue: string) => {
      const metadata = param.metadata || {};
      const min = metadata.min !== undefined ? parseFloat(metadata.min) : -Infinity;
      const max = metadata.max !== undefined ? parseFloat(metadata.max) : Infinity;
      const step = metadata.step !== undefined ? parseFloat(metadata.step) : 1;

      if (inputValue === '' || inputValue === '-') {
        return;
      }

      const numValue = parseFloat(inputValue);
      if (isNaN(numValue)) {
        return;
      }

      const hasMin = min !== -Infinity;
      const hasMax = max !== Infinity;
      let clampedValue = numValue;

      if (hasMin && numValue < min) {
        clampedValue = min;
        debouncedError('Invalid Value', `${param.name} must be at least ${min}. Value adjusted to ${min}.`);
      } else if (hasMax && numValue > max) {
        clampedValue = max;
        debouncedError('Invalid Value', `${param.name} must be at most ${max}. Value adjusted to ${max}.`);
      }

      if (clampedValue !== numValue) {
        handleParameterChange(filterId, paramId, clampedValue.toString());
      } else {
        handleParameterChange(filterId, paramId, numValue.toString());
      }
    },
    [handleParameterChange, debouncedError]
  );

  const handleDragStart = useCallback((index: number) => {
    setDraggedIndex(index);
  }, []);

  const handleDragEnd = useCallback(() => {
    setDraggedIndex(null);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback(
    (dropIndex: number) => {
      if (draggedIndex === null || draggedIndex === dropIndex) {
        return;
      }

      setLocalFilters((prev) => {
        const newFilters = [...prev];
        const [draggedFilter] = newFilters.splice(draggedIndex, 1);
        newFilters.splice(dropIndex, 0, draggedFilter);
        onFiltersChange(getActiveFilters(newFilters));
        return newFilters;
      });

      setDraggedIndex(null);
    },
    [draggedIndex, onFiltersChange]
  );

  const getActiveFilters = (filterStates: FilterState[]): ActiveFilterState[] => {
    const active = filterStates.filter((f) => f.enabled);
    if (active.length === 0) {
      return [{ id: 'none', parameters: {} }];
    }
    return active.map((filter) => ({
      id: filter.id,
      parameters: { ...filter.parameterValues },
    }));
  };

  const renderParameterControl = (filter: FilterState, param: GenericFilterParameter) => {
    const currentValue = filter.parameterValues[param.id] || param.defaultValue || '';

    switch (param.type) {
      case 1: // SELECT
        return (
          <div key={param.id} className={styles.paramControl}>
            <label className={styles.paramLabel}>{param.name}</label>
            <div className={styles.radioGroup}>
              {param.options?.map((option: GenericFilterParameterOption) => (
                <label key={option.value} className={styles.radioOption}>
                  <input
                    type="radio"
                    name={`${filter.id}-${param.id}`}
                    value={option.value}
                    checked={currentValue === option.value}
                    onChange={() => handleParameterChange(filter.id, param.id, option.value)}
                    data-testid={`filter-parameter-${filter.id}-${param.id}-${option.value}`}
                  />
                  <span>{option.label || option.value}</span>
                </label>
              ))}
            </div>
          </div>
        );

      case 2: // RANGE
        const metadata = param.metadata || {};
        const min = metadata.min !== undefined ? parseFloat(metadata.min) : 3;
        const max = metadata.max !== undefined ? parseFloat(metadata.max) : 15;
        const step = metadata.step !== undefined ? parseFloat(metadata.step) : 2;
        const rangeValue = currentValue ? parseFloat(currentValue) : (param.defaultValue ? parseFloat(param.defaultValue) : min);

        return (
          <div key={param.id} className={styles.paramControl}>
            <label className={styles.paramLabel}>{param.name}</label>
            <div className={styles.sliderContainer}>
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={rangeValue}
                onChange={(e) => handleParameterChange(filter.id, param.id, e.target.value)}
                data-testid={`filter-parameter-${filter.id}-${param.id}`}
              />
              <span className={styles.sliderValue}>{rangeValue}</span>
            </div>
          </div>
        );

      case 3: // NUMBER
        const numMetadata = param.metadata || {};
        const numMin = numMetadata.min !== undefined ? parseFloat(numMetadata.min) : -Infinity;
        const numMax = numMetadata.max !== undefined ? parseFloat(numMetadata.max) : Infinity;
        const numStep = numMetadata.step !== undefined ? parseFloat(numMetadata.step) : 1;
        const numValue = currentValue || param.defaultValue || (numMin !== -Infinity ? numMin : 0);

        return (
          <div key={param.id} className={styles.paramControl}>
            <label className={styles.paramLabel}>{param.name}</label>
            <div className={styles.numberInputContainer}>
              <input
                type="number"
                min={numMin !== -Infinity ? numMin : undefined}
                max={numMax !== Infinity ? numMax : undefined}
                step={numStep}
                value={numValue}
                onChange={(e) => handleNumberInputChange(filter.id, param.id, param, e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    const current = parseFloat(e.currentTarget.value) || 0;
                    const newValue = e.key === 'ArrowUp' ? current + numStep : current - numStep;
                    if ((numMin !== -Infinity && newValue < numMin) || (numMax !== Infinity && newValue > numMax)) {
                      return;
                    }
                    handleParameterChange(filter.id, param.id, newValue.toString());
                  }
                }}
                onWheel={(e) => {
                  e.preventDefault();
                  const current = parseFloat(e.currentTarget.value) || 0;
                  const delta = e.deltaY < 0 ? numStep : -numStep;
                  let newValue = current + delta;
                  newValue = Math.max(numMin, Math.min(numMax, newValue));
                  handleParameterChange(filter.id, param.id, newValue.toString());
                }}
                data-testid={`filter-parameter-${filter.id}-${param.id}`}
              />
            </div>
          </div>
        );

      case 4: // CHECKBOX
        const checked = currentValue === 'true' || currentValue === true || (param.defaultValue === 'true' && !currentValue);

        return (
          <div key={param.id} className={styles.paramControl}>
            <label className={styles.checkboxOption}>
              <input
                type="checkbox"
                checked={checked}
                onChange={(e) => handleParameterChange(filter.id, param.id, e.target.checked.toString())}
                data-testid={`filter-parameter-${filter.id}-${param.id}`}
              />
              <span>{param.name}</span>
            </label>
          </div>
        );

      default:
        return null;
    }
  };

  if (localFilters.length === 0) {
    return <div className={styles.noFilters}>No filters available</div>;
  }

  return (
    <div className={styles.controlSection}>
      <label className={styles.controlLabel}>
        Filters <span className={styles.hint}>(drag to reorder)</span>
      </label>
      <div className={styles.filtersList}>
        {localFilters.map((filter, index) => (
          <div
            key={filter.id}
            className={`${styles.filterCard} ${draggedIndex === index ? styles.dragging : ''}`}
            draggable
            onDragStart={() => handleDragStart(index)}
            onDragEnd={handleDragEnd}
            onDragOver={handleDragOver}
            onDrop={() => handleDrop(index)}
            data-filter-name={filter.name}
            data-filter-id={filter.id}
          >
            <div
              className={styles.filterHeader}
              onClick={() => toggleCard(index)}
            >
              <div
                className={styles.dragHandleContainer}
                onClick={(e) => e.stopPropagation()}
              >
                <span className={styles.dragHandle}>⋮⋮</span>
              </div>
              <div
                className={styles.checkboxContainer}
                onClick={(e) => e.stopPropagation()}
              >
                <input
                  type="checkbox"
                  checked={filter.enabled}
                  onChange={(e) => handleCheckboxChange(index, e.target.checked)}
                  onClick={(e) => e.stopPropagation()}
                  data-testid={`filter-checkbox-${filter.id}`}
                />
              </div>
              <label>{filter.name}</label>
              <span className={`${styles.chevron} ${filter.expanded ? styles.expanded : ''}`}>▶</span>
            </div>

            {filter.parameters.length > 0 && (
              <div className={`${styles.filterBody} ${filter.expanded ? styles.expanded : ''}`}>
                {filter.parameters.map((param) => renderParameterControl(filter, param))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
