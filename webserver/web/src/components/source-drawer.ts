import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { InputSource } from '../gen/config_service_pb';
import { telemetryService } from '../services/telemetry-service';

@customElement('source-drawer')
export class SourceDrawer extends LitElement {
    @state() private isOpen = false;
    @state() private availableSources: InputSource[] = [];
    @state() private selectedSourceIds: Set<string> = new Set();

    static styles = css`
        :host {
            position: fixed;
            top: 0;
            right: 0;
            bottom: 0;
            width: 100%;
            z-index: 9999;
            pointer-events: none;
        }

        .backdrop {
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.5);
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }

        .backdrop.show {
            opacity: 1;
            pointer-events: auto;
        }

        .drawer {
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 400px;
            max-width: 90vw;
            background: white;
            box-shadow: -4px 0 24px rgba(0, 0, 0, 0.15);
            transform: translateX(100%);
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            flex-direction: column;
            pointer-events: auto;
        }

        .drawer.show {
            transform: translateX(0);
        }

        .drawer-header {
            padding: 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .drawer-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .close-btn {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: var(--text-secondary);
            padding: 4px;
            line-height: 1;
            transition: color 0.2s;
        }

        .close-btn:hover {
            color: var(--text-primary);
        }

        .drawer-content {
            flex: 1;
            overflow-y: auto;
            padding: 16px 24px;
        }

        .source-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .source-item {
            padding: 16px;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .source-item:hover {
            border-color: var(--accent-color);
            background: var(--background-secondary);
        }

        .source-item.disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: var(--background-secondary);
        }

        .source-item.disabled:hover {
            border-color: var(--border-color);
        }

        .source-icon {
            font-size: 24px;
            flex-shrink: 0;
        }

        .source-info {
            flex: 1;
        }

        .source-name {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .source-type {
            font-size: 13px;
            color: var(--text-secondary);
            text-transform: capitalize;
        }

        .source-badge {
            background: var(--accent-color);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
    `;

    render() {
        return html`
            <div class="backdrop ${this.isOpen ? 'show' : ''}" @click=${this.close}></div>
            <div class="drawer ${this.isOpen ? 'show' : ''}">
                <div class="drawer-header">
                    <h2 class="drawer-title">Select Input Source</h2>
                    <button class="close-btn" @click=${this.close}>×</button>
                </div>
                <div class="drawer-content">
                    <div class="source-list">
                        ${this.availableSources.map(source => this.renderSourceItem(source))}
                    </div>
                </div>
            </div>
        `;
    }

    private renderSourceItem(source: InputSource) {
        const icon = source.type === 'camera' ? '●' : '▣';

        return html`
            <div 
                class="source-item"
                @click=${() => this.selectSource(source)}
            >
                <div class="source-icon">${icon}</div>
                <div class="source-info">
                    <div class="source-name">${source.displayName}</div>
                    <div class="source-type">${source.type}</div>
                </div>
                ${source.isDefault ? html`<span class="source-badge">Default</span>` : ''}
            </div>
        `;
    }

    open(sources: InputSource[], selectedIds: Set<string>): void {
        console.log('Drawer opened:', { available: sources.length, selected: selectedIds.size });
        
        this.availableSources = sources;
        this.selectedSourceIds = selectedIds;
        this.isOpen = true;
    }

    close(): void {
        this.isOpen = false;
    }

    private selectSource(source: InputSource): void {
        console.log('Source selected:', source.id, source.type);

        this.dispatchEvent(new CustomEvent('source-selected', {
            bubbles: true,
            composed: true,
            detail: { source }
        }));

        this.close();
    }
}

declare global {
    interface HTMLElementTagNameMap {
        'source-drawer': SourceDrawer;
    }
}

