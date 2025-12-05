import { LitElement, html, css } from 'lit';
import { customElement } from 'lit/decorators.js';
import { telemetryService } from '../../infrastructure/observability/telemetry-service';
import { logger } from '../../infrastructure/observability/otel-logger';

@customElement('add-source-fab')
export class AddSourceFab extends LitElement {
  static styles = css`
    :host {
      position: fixed;
      bottom: 90px;
      right: 32px;
      z-index: 1000;
    }

    .fab {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px 24px;
      background: var(--accent-color);
      color: white;
      border: none;
      border-radius: 28px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
      cursor: pointer;
      font-size: 16px;
      font-weight: 600;
      transition: all 0.2s;
    }

    .fab:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    }

    .fab:active {
      transform: translateY(0);
    }

    .fab-icon {
      font-size: 24px;
      line-height: 1;
    }

    @media (max-width: 768px) {
      .fab-text {
        display: none;
      }

      .fab {
        padding: 16px;
        border-radius: 50%;
        width: 56px;
        height: 56px;
        justify-content: center;
      }
    }
  `;

  render() {
    return html`
      <button class="fab" @click=${this.handleClick} data-testid="add-input-fab">
        <span class="fab-icon">+</span>
        <span class="fab-text">Add Input</span>
      </button>
    `;
  }

  private handleClick(): void {
    logger.debug('FAB clicked - opening drawer');

    this.dispatchEvent(
      new CustomEvent('open-drawer', {
        bubbles: true,
        composed: true,
      })
    );
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'add-source-fab': AddSourceFab;
  }
}
