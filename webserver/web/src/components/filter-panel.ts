import { LitElement, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { filterPanelStyles } from './filter-panel.styles';
import { Filter, GRAYSCALE_ALGORITHMS, DEFAULT_FILTERS } from './filter-panel.types';

@customElement('filter-panel')
export class FilterPanel extends LitElement {
    @state() private filters: Filter[] = DEFAULT_FILTERS;
    @state() private grayscaleAlgo = 'bt601';
    
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
        const disabled = filter.disabled;
        
        return html`
            <div 
                class="filter-card ${disabled ? 'disabled' : ''}"
                draggable="${!disabled}"
                @dragstart=${(e: DragEvent) => this.handleDragStart(e, index)}
                @dragend=${() => this.handleDragEnd()}
                @dragover=${(e: DragEvent) => this.handleDragOver(e)}
                @drop=${(e: DragEvent) => this.handleDrop(e, index)}
                @dragenter=${(e: Event) => this.handleDragEnter(e)}
                @dragleave=${(e: Event) => this.handleDragLeave(e)}
            >
                <div class="filter-header" @click=${() => !disabled && this.toggleCard(index)}>
                    <span class="drag-handle ${disabled ? 'disabled' : ''}">⋮⋮</span>
                    <input 
                        type="checkbox" 
                        .checked=${filter.enabled}
                        ?disabled=${disabled}
                        @change=${(e: Event) => this.handleCheckboxChange(index, e)}
                        @click=${(e: Event) => e.stopPropagation()}
                    />
                    <label @click=${(e: Event) => e.stopPropagation()}>
                        ${filter.name}
                        ${disabled ? html`<span class="badge">Soon</span>` : ''}
                    </label>
                    <span class="chevron ${filter.expanded ? 'expanded' : ''}">▶</span>
                </div>
                
                ${filter.id === 'grayscale' && !disabled ? html`
                    <div class="filter-body ${filter.expanded ? 'expanded' : ''}">
                        <label class="radio-label">Algorithm</label>
                        <div class="radio-group">
                            ${this.renderGrayscaleOptions()}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    private renderGrayscaleOptions() {
        return GRAYSCALE_ALGORITHMS.map(opt => html`
            <label class="radio-option">
                <input 
                    type="radio" 
                    name="grayscale-algo" 
                    value="${opt.value}"
                    .checked=${this.grayscaleAlgo === opt.value}
                    @change=${() => this.handleAlgoChange(opt.value)}
                />
                <span>${opt.label}</span>
            </label>
        `);
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

    private handleAlgoChange(value: string) {
        this.grayscaleAlgo = value;
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
        if (!target.classList.contains('disabled')) {
            target.classList.add('drag-over');
        }
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
        this.dispatchEvent(new CustomEvent('filter-change', {
            bubbles: true,
            composed: true,
            detail: {
                filters: this.getSelectedFilters(),
                grayscaleType: this.grayscaleAlgo
            }
        }));
    }

    getSelectedFilters(): string[] {
        const selected = this.filters
            .filter(f => f.enabled && !f.disabled)
            .map(f => f.id);
        return selected.length > 0 ? selected : ['none'];
    }

    getGrayscaleType(): string {
        return this.grayscaleAlgo;
    }

    setFilters(filters: string[], grayscaleType: string) {
        this.filters = this.filters.map(f => ({
            ...f,
            enabled: filters.includes(f.id),
            expanded: filters.includes(f.id)
        }));
        this.grayscaleAlgo = grayscaleType;
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

