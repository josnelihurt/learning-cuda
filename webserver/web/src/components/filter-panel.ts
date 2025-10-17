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
                <label class="control-label">
                    Filters <span class="hint">(drag to reorder)</span>
                </label>
                
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
            >
                <div class="filter-header" @click=${() => this.toggleCard(index)}>
                    <span class="drag-handle">⋮⋮</span>
                    <input 
                        type="checkbox" 
                        .checked=${filter.enabled}
                        @change=${(e: Event) => this.handleCheckboxChange(index, e)}
                        @click=${(e: Event) => e.stopPropagation()}
                    />
                    <label @click=${(e: Event) => e.stopPropagation()}>
                        ${filter.name}
                    </label>
                    <span class="chevron ${filter.expanded ? 'expanded' : ''}">▶</span>
                </div>
                
                ${filter.parameters.length > 0 ? html`
                    <div class="filter-body ${filter.expanded ? 'expanded' : ''}">
                        ${filter.parameters.map(param => this.renderParameter(filter, param))}
                    </div>
                ` : ''}
            </div>
        `;
    }

    private renderParameter(filter: Filter, param: any) {
        if (param.type === 'select') {
            return html`
                <label class="radio-label">${param.name}</label>
                <div class="radio-group">
                    ${param.options.map((option: string) => html`
                        <label class="radio-option">
                            <input 
                                type="radio" 
                                name="${filter.id}-${param.id}" 
                                value="${option}"
                                .checked=${filter.parameterValues[param.id] === option}
                                @change=${() => this.handleParameterChange(filter.id, param.id, option)}
                            />
                            <span>${formatParameterLabel(param, option)}</span>
                        </label>
                    `)}
                </div>
            `;
        }
        return html``;
    }

    private toggleCard(index: number) {
        this.filters = this.filters.map((f, i) => 
            i === index ? { ...f, expanded: !f.expanded } : f
        );
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
        this.filters = this.filters.map(f => {
            if (f.id === filterId) {
                return {
                    ...f,
                    parameterValues: {
                        ...f.parameterValues,
                        [paramId]: value
                    }
                };
            }
            return f;
        });
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
        cards?.forEach(card => {
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

        console.log('Filter order:', this.filters.map(f => f.id));
    }

    private dispatchFilterChange() {
        const grayscaleFilter = this.filters.find(f => f.id === 'grayscale');
        const grayscaleType = grayscaleFilter?.parameterValues['algorithm'] || 'bt601';
        
        this.dispatchEvent(new CustomEvent('filter-change', {
            bubbles: true,
            composed: true,
            detail: {
                filters: this.getSelectedFilters(),
                grayscaleType: grayscaleType
            }
        }));
    }

    getSelectedFilters(): string[] {
        const selected = this.filters
            .filter(f => f.enabled)
            .map(f => f.id);
        return selected.length > 0 ? selected : ['none'];
    }

    getGrayscaleType(): string {
        const grayscaleFilter = this.filters.find(f => f.id === 'grayscale');
        return grayscaleFilter?.parameterValues['algorithm'] || 'bt601';
    }

    setFilters(filters: string[], grayscaleType: string) {
        this.filters = this.filters.map(f => {
            const enabled = filters.includes(f.id);
            const parameterValues = { ...f.parameterValues };
            
            if (f.id === 'grayscale' && grayscaleType) {
                parameterValues['algorithm'] = grayscaleType;
            }
            
            return {
                ...f,
                enabled,
                expanded: enabled,
                parameterValues
            };
        });
        this.requestUpdate();
    }

    updateFiltersUI() {
        this.filters = this.filters.map(f => ({
            ...f,
            expanded: f.enabled
        }));
    }
}

declare global {
    interface HTMLElementTagNameMap {
        'filter-panel': FilterPanel;
    }
}

