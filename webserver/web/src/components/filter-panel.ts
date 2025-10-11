import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';

interface Filter {
    id: string;
    name: string;
    enabled: boolean;
    expanded: boolean;
    disabled?: boolean;
}

@customElement('filter-panel')
export class FilterPanel extends LitElement {
    @state() private filters: Filter[] = [
        { id: 'grayscale', name: 'Grayscale', enabled: false, expanded: false },
        { id: 'blur', name: 'Blur', enabled: false, expanded: false, disabled: true },
        { id: 'edge', name: 'Edge Detect', enabled: false, expanded: false, disabled: true }
    ];

    @state() private grayscaleAlgo = 'bt601';
    
    private draggedIndex: number | null = null;

    static styles = css`
        :host {
            display: block;
        }

        .control-section {
            margin-bottom: 24px;
        }

        .control-label {
            display: block;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #333;
            margin-bottom: 12px;
        }

        .hint {
            font-weight: 400;
            font-size: 11px;
            color: #999;
            text-transform: none;
            letter-spacing: 0;
        }

        .filters-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .filter-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            overflow: hidden;
            transition: all 0.2s;
        }

        .filter-card:not(.disabled) {
            cursor: move;
        }

        .filter-card.dragging {
            opacity: 0.5;
        }

        .filter-card.drag-over {
            border-color: #ffa400;
            box-shadow: 0 0 0 2px rgba(255, 164, 0, 0.1);
        }

        .filter-card.disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .filter-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            cursor: pointer;
            user-select: none;
        }

        .filter-card.disabled .filter-header {
            cursor: not-allowed;
        }

        .drag-handle {
            color: #999;
            font-size: 12px;
            cursor: move;
        }

        .drag-handle.disabled {
            cursor: not-allowed;
        }

        input[type="checkbox"] {
            cursor: pointer;
        }

        .filter-header label {
            flex: 1;
            cursor: pointer;
            font-weight: 500;
        }

        .badge {
            background: #ffa400;
            color: white;
            font-size: 9px;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
            margin-left: 6px;
        }

        .chevron {
            color: #666;
            font-size: 10px;
            transition: transform 0.2s;
        }

        .filter-card .chevron.expanded {
            transform: rotate(90deg);
        }

        .filter-body {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .filter-body.expanded {
            max-height: 400px;
            padding: 0 16px 16px 16px;
        }

        .radio-label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            color: #666;
            margin-bottom: 8px;
        }

        .radio-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .radio-option {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .radio-option:hover {
            background: #f5f5f5;
        }

        .radio-option input[type="radio"] {
            cursor: pointer;
        }

        .radio-option span {
            font-size: 13px;
            color: #333;
        }
    `;

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
        const options = [
            { value: 'bt601', label: 'ITU-R BT.601 (SDTV)' },
            { value: 'bt709', label: 'ITU-R BT.709 (HDTV)' },
            { value: 'average', label: 'Average' },
            { value: 'lightness', label: 'Lightness' },
            { value: 'luminosity', label: 'Luminosity' }
        ];

        return options.map(opt => html`
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

