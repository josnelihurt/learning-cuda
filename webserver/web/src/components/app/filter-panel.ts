import { LitElement, html } from 'lit';
import { customElement, state, property } from 'lit/decorators.js';
import { filterPanelStyles } from './filter-panel.styles';
import { Filter, formatParameterLabel, ActiveFilterState } from './filter-panel.types';
import { logger } from '../../infrastructure/observability/otel-logger';
import type { ToastContainer } from './toast-container';

@customElement('filter-panel')
export class FilterPanel extends LitElement {
  @property({ type: Array }) filters: Filter[] = [];

  private draggedIndex: number | null = null;

  static styles = filterPanelStyles;

  private getToastManager(): ToastContainer | null {
    const toastElement = document.querySelector('toast-container') as ToastContainer;
    return toastElement || null;
  }

  firstUpdated() {
    this.setupNumberInputListeners();
  }

  updated(changedProperties: Map<PropertyKey, unknown>) {
    super.updated(changedProperties);
    if (changedProperties.has('filters')) {
      logger.debug('FilterPanel filters updated', {
        'filters.count': this.filters.length,
        'filters': this.filters.map((f) => f.name).join(','),
      });
    }
    this.setupNumberInputListeners();
  }

  private setupNumberInputListeners() {
    const numberInputs = this.shadowRoot?.querySelectorAll('input[type="number"]');
    numberInputs?.forEach((input) => {
      const inputElement = input as HTMLInputElement;
      const min = inputElement.min ? parseFloat(inputElement.min) : -Infinity;
      const max = inputElement.max ? parseFloat(inputElement.max) : Infinity;
      const step = inputElement.step ? parseFloat(inputElement.step) : 1;
      const hasMin = min !== -Infinity;
      const hasMax = max !== Infinity;

      const handleClick = (e: Event) => {
        const mouseEvent = e as MouseEvent;
        const target = mouseEvent.target as HTMLElement;
        if (target === inputElement) {
          const rect = inputElement.getBoundingClientRect();
          const clickX = mouseEvent.clientX - rect.left;
          const inputWidth = rect.width;

          if (clickX > inputWidth - 40) {
            const currentValue = parseFloat(inputElement.value) || 0;
            const buttonType = clickX > inputWidth - 20 ? 'up' : 'down';
            const newValue = buttonType === 'up' ? currentValue + step : currentValue - step;

            if (buttonType === 'up' && hasMax && newValue > max) {
              mouseEvent.preventDefault();
              mouseEvent.stopPropagation();
              return false;
            }
            if (buttonType === 'down' && hasMin && newValue < min) {
              mouseEvent.preventDefault();
              mouseEvent.stopPropagation();
              return false;
            }
          }
        }
        return true;
      };

      inputElement.removeEventListener('mousedown', handleClick);
      inputElement.addEventListener('mousedown', handleClick, true);
    });
  }

  render() {
    return html`
      <div class="control-section filters-section">
        <label class="control-label"> Filters <span class="hint">(drag to reorder)</span> </label>

        <div class="filters-list">
          ${this.filters.length === 0
            ? html`<div class="no-filters">No filters available</div>`
            : this.filters.map((filter, index) => this.renderFilterCard(filter, index))}
        </div>
      </div>
    `;
  }

  private renderFilterCard(filter: Filter, index: number) {
    return html`
      <div
        class="filter-card"
        @dragend=${() => this.handleDragEnd()}
        @dragover=${(e: DragEvent) => this.handleDragOver(e)}
        @drop=${(e: DragEvent) => this.handleDrop(e, index)}
        @dragenter=${(e: Event) => this.handleDragEnter(e)}
        @dragleave=${(e: Event) => this.handleDragLeave(e)}
        data-filter-name="${filter.name}"
        data-filter-id="${filter.id}"
      >
        <div class="filter-header" @click=${() => this.toggleCard(index)}>
          <div 
            class="drag-handle-container"
            draggable="true"
            @dragstart=${(e: DragEvent) => this.handleDragStart(e, index)}
            @click=${(e: Event) => e.stopPropagation()}
          >
            <span class="drag-handle">⋮⋮</span>
          </div>
          <div 
            class="checkbox-container" 
            draggable="false"
            @click=${(e: Event) => {
              e.stopPropagation();
              const checkbox = (e.currentTarget as HTMLElement).querySelector('input[type="checkbox"]') as HTMLInputElement;
              if (checkbox && e.target !== checkbox) {
                checkbox.click();
              }
            }}
          >
            <input
              type="checkbox"
              .checked=${filter.enabled}
              draggable="false"
              @change=${(e: Event) => this.handleCheckboxChange(index, e)}
              @click=${(e: Event) => e.stopPropagation()}
              data-testid="filter-checkbox-${filter.id}"
            />
          </div>
          <label> ${filter.name} </label>
          <span class="chevron ${filter.expanded ? 'expanded' : ''}">▶</span>
        </div>

        ${filter.parameters.length > 0
          ? html`
              <div class="filter-body ${filter.expanded ? 'expanded' : ''}">
                ${filter.parameters.map((param) => this.renderParameter(filter, param))}
              </div>
            `
          : ''}
      </div>
    `;
  }

  private renderParameter(filter: Filter, param: any) {
    const currentValue = filter.parameterValues[param.id] || param.defaultValue || '';

    if (param.type === 'select') {
      const options = param.options || [];
      if (options.length === 0) {
        logger.warn('Filter parameter has no options', {
          'filter.id': filter.id,
          'param.id': param.id,
          'param.name': param.name,
        });
        return html`
          <div class="param-control">
            <label class="param-label">${param.name}</label>
            <div class="no-options">No options available</div>
          </div>
        `;
      }
      return html`
        <div class="param-control">
          <label class="param-label">${param.name}</label>
          <div class="radio-group">
            ${options.map(
              (option: { value: string; label: string }) => html`
                <label class="radio-option">
                  <input
                    type="radio"
                    name="${filter.id}-${param.id}"
                    value="${option.value}"
                    .checked=${currentValue === option.value}
                    draggable="false"
                    @change=${() => this.handleParameterChange(filter.id, param.id, option.value)}
                    data-testid="filter-parameter-${filter.id}-${param.id}-${option.value}"
                  />
                  <span>${formatParameterLabel(param, option.value)}</span>
                </label>
              `
            )}
          </div>
        </div>
      `;
    }

    if (param.type === 'slider' || param.type === 'range') {
      const min = param.min !== undefined ? param.min : 3;
      const max = param.max !== undefined ? param.max : 15;
      const step = param.step !== undefined ? param.step : 2;
      const value = currentValue ? parseFloat(currentValue) : (param.defaultValue ? parseFloat(param.defaultValue) : min);

      return html`
        <div class="param-control">
          <label class="param-label">${param.name}</label>
          <div class="slider-container">
            <input
              type="range"
              min="${min}"
              max="${max}"
              step="${step}"
              .value="${value.toString()}"
              draggable="false"
              @input=${(e: Event) => {
                const newValue = (e.target as HTMLInputElement).value;
                this.handleParameterChange(filter.id, param.id, newValue);
              }}
              data-testid="filter-parameter-${filter.id}-${param.id}"
            />
            <span class="slider-value">${value}</span>
          </div>
        </div>
      `;
    }

    if (param.type === 'number') {
      const min = param.min !== undefined ? param.min : -Infinity;
      const max = param.max !== undefined ? param.max : Infinity;
      const step = param.step !== undefined ? param.step : 1;
      const numericValue = currentValue ? parseFloat(currentValue) : (param.defaultValue ? parseFloat(param.defaultValue) : (min !== -Infinity ? min : 0));
      const clampedValue = Math.max(min, Math.min(max, isNaN(numericValue) ? (min !== -Infinity ? min : 0) : numericValue));
      const displayValue = currentValue || param.defaultValue || clampedValue.toString();

      return html`
        <div class="param-control">
          <label class="param-label">${param.name}</label>
          <div class="number-input-container">
            <input
              type="number"
              min="${min !== -Infinity ? min : ''}"
              max="${max !== Infinity ? max : ''}"
              step="${step}"
              .value="${displayValue}"
              draggable="false"
              @change=${(e: Event) => {
                this.handleNumberInputChange(filter.id, param.id, e, min, max, param.name);
              }}
              @input=${(e: Event) => {
                const input = e.target as HTMLInputElement;
                const inputValue = input.value.trim();
                if (inputValue === '' || inputValue === '-') {
                  return;
                }
                const numValue = parseFloat(inputValue);
                if (isNaN(numValue)) {
                  return;
                }
                
                const hasMin = min !== -Infinity;
                const hasMax = max !== Infinity;
                let needsClamp = false;
                let clampedValue = numValue;
                
                if (hasMin && numValue < min) {
                  clampedValue = min;
                  needsClamp = true;
                } else if (hasMax && numValue > max) {
                  clampedValue = max;
                  needsClamp = true;
                }
                
                if (needsClamp) {
                  input.value = clampedValue.toString();
                  this.handleParameterChange(filter.id, param.id, clampedValue.toString());
                  const toastManager = this.getToastManager();
                  if (toastManager) {
                    const paramName = param.name || 'Value';
                    if (hasMin && numValue < min) {
                      toastManager.error('Invalid Value', `${paramName} must be at least ${min}. Value adjusted to ${min}.`);
                    } else if (hasMax && numValue > max) {
                      toastManager.error('Invalid Value', `${paramName} must be at most ${max}. Value adjusted to ${max}.`);
                    }
                  }
                } else {
                  this.handleParameterChange(filter.id, param.id, numValue.toString());
                }
              }}
              @keydown=${(e: KeyboardEvent) => {
                if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
                  const input = e.target as HTMLInputElement;
                  const currentValue = parseFloat(input.value) || 0;
                  const stepValue = step;
                  const newValue = e.key === 'ArrowUp' ? currentValue + stepValue : currentValue - stepValue;
                  const hasMin = min !== -Infinity;
                  const hasMax = max !== Infinity;
                  
                  if (e.key === 'ArrowUp' && hasMax && newValue > max) {
                    e.preventDefault();
                    return;
                  }
                  if (e.key === 'ArrowDown' && hasMin && newValue < min) {
                    e.preventDefault();
                    return;
                  }
                }
              }}
              @wheel=${(e: WheelEvent) => {
                const input = e.target as HTMLInputElement;
                if (document.activeElement === input) {
                  e.preventDefault();
                  const currentValue = parseFloat(input.value) || 0;
                  const stepValue = step;
                  const hasMin = min !== -Infinity;
                  const hasMax = max !== Infinity;
                  const newValue = e.deltaY < 0 ? currentValue + stepValue : currentValue - stepValue;
                  let clampedValue = newValue;
                  
                  if (hasMin) {
                    clampedValue = Math.max(min, clampedValue);
                  }
                  if (hasMax) {
                    clampedValue = Math.min(max, clampedValue);
                  }
                  
                  if (clampedValue !== currentValue) {
                    input.value = clampedValue.toString();
                    this.handleParameterChange(filter.id, param.id, clampedValue.toString());
                  }
                }
              }}
              data-testid="filter-parameter-${filter.id}-${param.id}"
            />
          </div>
        </div>
      `;
    }

    if (param.type === 'checkbox') {
      const checked =
        currentValue === 'true' ||
        currentValue === true ||
        (param.defaultValue === 'true' && !currentValue);

      return html`
        <div class="param-control">
          <label class="checkbox-option">
            <input
              type="checkbox"
              .checked=${checked}
              draggable="false"
              @change=${(e: Event) => {
                const checked = (e.target as HTMLInputElement).checked;
                this.handleParameterChange(filter.id, param.id, checked.toString());
              }}
              data-testid="filter-parameter-${filter.id}-${param.id}"
            />
            <span>${param.name}</span>
          </label>
        </div>
      `;
    }

    return html``;
  }

  private toggleCard(index: number) {
    this.filters = this.filters.map((f, i) => {
      if (i === index) {
        if (!f.expanded) {
          return {
            ...f,
            expanded: true,
            enabled: !f.enabled ? true : f.enabled,
          };
        } else {
          return { ...f, expanded: false };
        }
      }
      return f;
    });
    this.dispatchFilterChange();
  }

  private handleCheckboxChange(index: number, e: Event) {
    const checked = (e.target as HTMLInputElement).checked;
    this.filters = this.filters.map((f, i) => {
      if (i === index) {
        return { 
          ...f, 
          enabled: checked,
          expanded: checked ? f.expanded : false,
        };
      }
      return f;
    });
    this.dispatchFilterChange();
  }

  private handleNumberInputChange(
    filterId: string,
    paramId: string,
    e: Event,
    min: number,
    max: number,
    paramName: string
  ) {
    const input = e.target as HTMLInputElement;
    let inputValue = input.value.trim();

    if (inputValue === '' || inputValue === '-') {
      return;
    }

    const numValue = parseFloat(inputValue);
    let finalValue = numValue;
    let showError = false;
    let errorMessage = '';

    const hasMin = min !== -Infinity;
    const hasMax = max !== Infinity;

    if (isNaN(numValue)) {
      const filter = this.filters.find((f) => f.id === filterId);
      const currentValue = filter?.parameterValues[paramId] || '';
      finalValue = currentValue ? parseFloat(currentValue) : (hasMin ? min : 0);
      inputValue = finalValue.toString();
      showError = true;
      errorMessage = `Invalid number. Please enter a valid number.`;
    } else {
      if (hasMin && numValue < min) {
        finalValue = min;
        showError = true;
        errorMessage = `${paramName} must be at least ${min}. Value adjusted to ${min}.`;
      } else if (hasMax && numValue > max) {
        finalValue = max;
        showError = true;
        errorMessage = `${paramName} must be at most ${max}. Value adjusted to ${max}.`;
      }
    }

    if (finalValue !== numValue || input.value !== finalValue.toString()) {
      input.value = finalValue.toString();
    }

    this.handleParameterChange(filterId, paramId, finalValue.toString());

    if (showError) {
      const toastManager = this.getToastManager();
      if (toastManager) {
        toastManager.error('Invalid Value', errorMessage);
      }
    }

    this.requestUpdate();
  }

  private handleParameterChange(filterId: string, paramId: string, value: string) {
    this.filters = this.filters.map((f) => {
      if (f.id === filterId) {
        return {
          ...f,
          parameterValues: {
            ...f.parameterValues,
            [paramId]: value,
          },
        };
      }
      return f;
    });
    this.requestUpdate();
    this.dispatchFilterChange();
  }

  private handleDragStart(e: DragEvent, index: number) {
    e.stopPropagation();
    this.draggedIndex = index;
    const card = (e.currentTarget as HTMLElement).closest('.filter-card') as HTMLElement;
    if (card) {
      card.classList.add('dragging');
    }
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = 'move';
    }
  }

  private handleDragEnd() {
    const cards = this.shadowRoot?.querySelectorAll('.filter-card');
    cards?.forEach((card) => {
      card.classList.remove('dragging');
      card.classList.remove('drag-over');
    });
  }

  private handleDragOver(e: DragEvent) {
    e.preventDefault();
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = 'move';
    }
  }

  private handleDragEnter(e: Event) {
    const target = e.currentTarget as HTMLElement;
    target.classList.add('drag-over');
  }

  private handleDragLeave(e: Event) {
    (e.currentTarget as HTMLElement).classList.remove('drag-over');
  }

  private handleDrop(e: DragEvent, dropIndex: number) {
    e.preventDefault();
    e.stopPropagation();

    if (this.draggedIndex === null || this.draggedIndex === dropIndex) return;

    const newFilters = [...this.filters];
    const [draggedFilter] = newFilters.splice(this.draggedIndex, 1);
    newFilters.splice(dropIndex, 0, draggedFilter);

    this.filters = newFilters;
    this.draggedIndex = null;

    logger.debug('Filter order', {
      filters: this.filters.map((f) => f.id).join(','),
    });
  }

  private dispatchFilterChange() {
    this.dispatchEvent(
      new CustomEvent('filter-change', {
        bubbles: true,
        composed: true,
        detail: {
          filters: this.getActiveFilters(),
        },
      })
    );
  }

  getActiveFilters(): ActiveFilterState[] {
    const active = this.filters.filter((f) => f.enabled);
    if (active.length === 0) {
      return [{ id: 'none', parameters: {} }];
    }
    return active.map((filter) => ({
      id: filter.id,
      parameters: { ...filter.parameterValues },
    }));
  }

  setFilters(filters: ActiveFilterState[]) {
    this.filters = this.filters.map((f) => {
      const nextState = filters.find((state) => state.id === f.id);
      const enabled = Boolean(nextState);
      const parameterValues = nextState ? { ...f.parameterValues, ...nextState.parameters } : { ...f.parameterValues };

      return {
        ...f,
        enabled,
        expanded: enabled,
        parameterValues,
      };
    });
    this.requestUpdate();
  }

  updateFiltersUI() {
    this.filters = this.filters.map((f) => ({
      ...f,
      expanded: f.enabled,
    }));
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'filter-panel': FilterPanel;
  }
}
