import { LitElement, html } from 'lit';
import { customElement, state, property } from 'lit/decorators.js';
import { filterPanelStyles } from './filter-panel.styles';
import { Filter, formatParameterLabel } from './filter-panel.types';

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
    const currentValue = filter.parameterValues[param.id] || param.default_value || '';

    if (param.type === 'select') {
      return html`
        <div class="param-control">
          <label class="param-label">${param.name}</label>
          <div class="radio-group">
            ${param.options.map(
              (option: string) => html`
                <label class="radio-option">
                  <input
                    type="radio"
                    name="${filter.id}-${param.id}"
                    value="${option}"
                    .checked=${currentValue === option}
                    @change=${() => this.handleParameterChange(filter.id, param.id, option)}
                    data-testid="filter-parameter-${filter.id}-${param.id}-${option}"
                  />
                  <span>${formatParameterLabel(param, option)}</span>
                </label>
              `
            )}
          </div>
        </div>
      `;
    }

    if (param.type === 'slider' || param.type === 'range') {
      const min = param.min ? parseInt(param.min) : 3;
      const max = param.max ? parseInt(param.max) : 15;
      const step = param.step ? parseInt(param.step) : 2;
      const value = currentValue ? parseInt(currentValue) : (param.default_value ? parseInt(param.default_value) : min);

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
      const min = param.min !== undefined ? parseFloat(param.min) : undefined;
      const max = param.max !== undefined ? parseFloat(param.max) : undefined;
      const step = param.step !== undefined ? parseFloat(param.step) : undefined;
      const value = currentValue || param.default_value || '';

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
      const checked = currentValue === 'true' || currentValue === true || (param.default_value === 'true' && !currentValue);

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
    const grayscaleFilter = this.filters.find((f) => f.id === 'grayscale');
    const grayscaleType = grayscaleFilter?.parameterValues['algorithm'] || 'bt601';

    const blurParams = this.getBlurParams();

    this.dispatchEvent(
      new CustomEvent('filter-change', {
        bubbles: true,
        composed: true,
        detail: {
          filters: this.getSelectedFilters(),
          grayscaleType: grayscaleType,
          blurParams: blurParams,
        },
      })
    );
  }

  getSelectedFilters(): string[] {
    const selected = this.filters.filter((f) => f.enabled).map((f) => f.id);
    return selected.length > 0 ? selected : ['none'];
  }

  getGrayscaleType(): string {
    const grayscaleFilter = this.filters.find((f) => f.id === 'grayscale');
    return grayscaleFilter?.parameterValues['algorithm'] || 'bt601';
  }

  getBlurParams(): Record<string, any> | undefined {
    const blurFilter = this.filters.find((f) => f.id === 'blur');
    if (!blurFilter || !blurFilter.enabled) {
      return undefined;
    }

    const params = blurFilter.parameterValues;
    const result: Record<string, any> = {};

    if (params.kernel_size !== undefined) {
      result.kernel_size = parseInt(params.kernel_size);
    }
    if (params.sigma !== undefined) {
      result.sigma = parseFloat(params.sigma);
    }
    if (params.border_mode !== undefined) {
      result.border_mode = params.border_mode;
    }
    if (params.separable !== undefined) {
      result.separable = params.separable === 'true';
    }

    return Object.keys(result).length > 0 ? result : undefined;
  }

  setFilters(filters: string[], grayscaleType: string, blurParams?: Record<string, any>) {
    this.filters = this.filters.map((f) => {
      const enabled = filters.includes(f.id);
      const parameterValues = { ...f.parameterValues };

      if (f.id === 'grayscale' && grayscaleType) {
        parameterValues['algorithm'] = grayscaleType;
      }

      if (f.id === 'blur' && blurParams) {
        if (blurParams.kernel_size !== undefined) {
          parameterValues['kernel_size'] = blurParams.kernel_size.toString();
        }
        if (blurParams.sigma !== undefined) {
          parameterValues['sigma'] = blurParams.sigma.toString();
        }
        if (blurParams.border_mode !== undefined) {
          parameterValues['border_mode'] = blurParams.border_mode;
        }
        if (blurParams.separable !== undefined) {
          parameterValues['separable'] = blurParams.separable.toString();
        }
      }

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
