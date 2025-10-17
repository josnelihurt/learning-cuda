import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { telemetryService } from '../services/telemetry-service';

@customElement('video-source-card')
export class VideoSourceCard extends LitElement {
    @property({ type: String }) sourceId = '';
    @property({ type: Number }) sourceNumber = 1;
    @property({ type: String }) sourceName = '';
    @property({ type: String }) sourceType = '';
    @property({ type: Boolean }) isSelected = false;

    static styles = css`
        :host {
            display: block;
            position: relative;
            width: 100%;
            height: 100%;
            min-height: 0;
        }

        .card {
            position: relative;
            width: 100%;
            height: 100%;
            border-radius: 8px;
            overflow: hidden;
            background: var(--background-secondary);
            border: 3px solid transparent;
            transition: all 0.2s;
            cursor: pointer;
        }

        .card:hover {
            border-color: var(--border-color);
        }

        .card.selected {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 1px var(--accent-color);
        }

        .source-number {
            position: absolute;
            top: 8px;
            left: 8px;
            background: rgba(0, 0, 0, 0.6);
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 600;
            z-index: 10;
        }

        .close-btn {
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0, 0, 0, 0.6);
            color: white;
            border: none;
            width: 28px;
            height: 28px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            cursor: pointer;
            opacity: 0;
            transition: all 0.2s;
            z-index: 10;
        }

        .card:hover .close-btn {
            opacity: 1;
        }

        .close-btn:hover {
            background: rgba(244, 67, 54, 0.9);
        }

        .content {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        ::slotted(img),
        ::slotted(video),
        ::slotted(camera-preview) {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
    `;

    render() {
        return html`
            <div 
                class="card ${this.isSelected ? 'selected' : ''}"
                @click=${this.handleCardClick}
                data-testid="source-card-${this.sourceNumber}"
                data-source-number="${this.sourceNumber}"
                data-source-id="${this.sourceId}"
            >
                <div class="source-number">${this.sourceNumber}</div>
                <button 
                    class="close-btn" 
                    @click=${this.handleClose}
                    title="Close ${this.sourceName}"
                    data-testid="source-close-button"
                >×</button>
                <div class="content">
                    <slot></slot>
                </div>
            </div>
        `;
    }

    private handleCardClick(e: Event): void {
        e.stopPropagation();
        
        console.log('Card selected:', this.sourceId, this.sourceNumber);

        this.dispatchEvent(new CustomEvent('source-selected', {
            bubbles: true,
            composed: true,
            detail: { sourceId: this.sourceId, sourceNumber: this.sourceNumber }
        }));
    }

    private handleClose(e: Event): void {
        e.stopPropagation();
        
        console.log('Card closed:', this.sourceId, this.sourceNumber);

        this.dispatchEvent(new CustomEvent('source-closed', {
            bubbles: true,
            composed: true,
            detail: { sourceId: this.sourceId, sourceNumber: this.sourceNumber }
        }));
    }
}

declare global {
    interface HTMLElementTagNameMap {
        'video-source-card': VideoSourceCard;
    }
}

