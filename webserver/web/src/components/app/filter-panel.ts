import { LitElement, html } from 'lit';
import { customElement, state, property } from 'lit/decorators.js';
import { filterPanelStyles } from './filter-panel.styles';
import { Filter, formatParameterLabel, ActiveFilterState } from './filter-panel.types';

@customElement('filter-panel')
export class FilterPanel extends LitElement {
  @property({ type: Array }) filters: Filter[] = [];

  private draggedIndex: number | null = null;

  static styles = filterPanelStyles;

  render() {
    return html`
      <div class="control-section filters-section">
        <label class="control-label"> Filters <span class="hint">(drag to reorder)</span> </label>

        <div class="filters-list">
          ${this.filters.map((filter, index) => this.renderFilterCard(filter, index))}
        </div>
      </div>
    `;
  }

  private renderFilterCard(filter: Filter, index: number) {
    return html`
      <div
        class="filter-card"
        draggable="true"
        @dragstart=${(e: DragEvent) => this.handleDragStart(e, index)}
        @dragend=${() => this.handleDragEnd()}
        @dragover=${(e: DragEvent) => this.handleDragOver(e)}
        @drop=${(e: DragEvent) => this.handleDrop(e, index)}
        @dragenter=${(e: Event) => this.handleDragEnter(e)}
        @dragleave=${(e: Event) => this.handleDragLeave(e)}
        data-filter-name="${filter.name}"
        data-filter-id="${filter.id}"
      >
        <div class="filter-header" @click=${() => this.toggleCard(index)}>
          <span class="drag-handle">⋮⋮</span>
          <input
            type="checkbox"
            .checked=${filter.enabled}
            @change=${(e: Event) => this.handleCheckboxChange(index, e)}
            @click=${(e: Event) => e.stopPropagation()}
            data-testid="filter-checkbox-${filter.id}"
          />
          <label @click=${(e: Event) => e.stopPropagation()}> ${filter.name} </label>
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
      return html`
        <div class="param-control">
          <label class="param-label">${param.name}</label>
          <div class="radio-group">
            ${param.options.map(
              (option: { value: string; label: string }) => html`
                <label class="radio-option">
                  <input
                    type="radio"
                    name="${filter.id}-${param.id}"
                    value="${option.value}"
                    .checked=${currentValue === option.value}
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
      const min = param.min;
      const max = param.max;
      const step = param.step;
      const value = currentValue || param.defaultValue || '';

      return html`
        <div class="param-control">
          <label class="param-label">${param.name}</label>
          <input
            type="number"
            min="${min !== undefined ? min : ''}"
            max="${max !== undefined ? max : ''}"
            step="${step !== undefined ? step : ''}"
            .value="${value}"
            @change=${(e: Event) => {
              const newValue = (e.target as HTMLInputElement).value;
              this.handleParameterChange(filter.id, param.id, newValue);
            }}
            @input=${(e: Event) => {
              const newValue = (e.target as HTMLInputElement).value;
              this.handleParameterChange(filter.id, param.id, newValue);
            }}
            data-testid="filter-parameter-${filter.id}-${param.id}"
          />
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
    this.filters = this.filters.map((f, i) => (i === index ? { ...f, expanded: !f.expanded } : f));
  }

  private handleCheckboxChange(index: number, e: Event) {
    const checked = (e.target as HTMLInputElement).checked;
    this.filters = this.filters.map((f, i) => {
      if (i === index) {
        return { ...f, enabled: checked, expanded: checked };
      }
      return f;
    });
    this.dispatchFilterChange();
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
    this.draggedIndex = index;
    (e.currentTarget as HTMLElement).classList.add('dragging');
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
