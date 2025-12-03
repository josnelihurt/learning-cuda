import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { StaticImage } from '../../gen/common_pb';
import { telemetryService } from '../../infrastructure/observability/telemetry-service';
import { logger } from '../../infrastructure/observability/otel-logger';

@customElement('image-selector-modal')
export class ImageSelectorModal extends LitElement {
  @state() private isOpen = false;
  @state() private availableImages: StaticImage[] = [];

  static styles = css`
    :host {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 10000;
      pointer-events: none;
      display: none;
    }

    :host([open]) {
      display: block;
    }

    .backdrop {
      position: absolute;
      inset: 0;
      background: rgba(0, 0, 0, 0.5);
      opacity: 0;
      transition: opacity 0.3s ease;
      pointer-events: none;
      z-index: 1;
    }

    .backdrop.show {
      opacity: 1;
      pointer-events: auto;
    }

    .modal {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) scale(0.9);
      width: 600px;
      max-width: 90vw;
      max-height: 80vh;
      background: white;
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
      opacity: 0;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      flex-direction: column;
      pointer-events: auto;
      z-index: 2;
    }

    .modal.show {
      opacity: 1;
      transform: translate(-50%, -50%) scale(1);
    }

    .modal-header {
      padding: 24px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .modal-title {
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

    .modal-content {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
    }

    .image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 16px;
    }

    .image-item {
      padding: 12px;
      border: 2px solid var(--border-color);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      background: var(--background-secondary);
    }

    .image-item:hover {
      border-color: var(--accent-color);
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .image-preview {
      width: 120px;
      height: 120px;
      object-fit: cover;
      border-radius: 4px;
      background: #f5f5f5;
    }

    .image-name {
      font-weight: 500;
      color: var(--text-primary);
      text-align: center;
      font-size: 14px;
    }

    .image-badge {
      background: var(--accent-color);
      color: white;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
    }

    .empty-state {
      text-align: center;
      padding: 40px 20px;
      color: var(--text-secondary);
    }
  `;

  render() {
    return html`
      <div class="backdrop ${this.isOpen ? 'show' : ''}" @click=${this.close}></div>
      <div class="modal ${this.isOpen ? 'show' : ''}" data-testid="image-selector-modal">
        <div class="modal-header">
          <h2 class="modal-title">Select Image</h2>
          <button class="close-btn" @click=${this.close} data-testid="modal-close">×</button>
        </div>
        <div class="modal-content">
          ${this.availableImages.length > 0
            ? html`<div class="image-grid">
                ${this.availableImages.map((image) => this.renderImageItem(image))}
              </div>`
            : html`<div class="empty-state">No images available</div>`}
        </div>
      </div>
    `;
  }

  private renderImageItem(image: StaticImage) {
    return html`
      <div
        class="image-item"
        @click=${() => this.selectImage(image)}
        data-testid="image-item-${image.id}"
      >
        <img src="${image.path}" alt="${image.displayName}" class="image-preview" loading="lazy" />
        <div class="image-name">${image.displayName}</div>
        ${image.isDefault ? html`<span class="image-badge">Default</span>` : ''}
      </div>
    `;
  }

  open(images: StaticImage[]): void {
    logger.debug('Image selector opened', {
      'images.available': images.length,
    });

    this.availableImages = images;
    this.isOpen = true;
    this.setAttribute('open', '');
  }

  close(): void {
    this.isOpen = false;
    this.removeAttribute('open');
  }

  private selectImage(image: StaticImage): void {
    logger.debug(`Image selected: ${image.id}`, {
      'image.id': image.id,
    });

    this.dispatchEvent(
      new CustomEvent('image-selected', {
        bubbles: true,
        composed: true,
        detail: { image },
      })
    );

    this.close();
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'image-selector-modal': ImageSelectorModal;
  }
}
