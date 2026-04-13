import { LitElement, html, css } from 'lit';
import { customElement } from 'lit/decorators.js';

@customElement('feature-flags-button')
export class FeatureFlagsButton extends LitElement {
  static styles = css`
    :host {
      display: inline-block;
    }

    .feature-flags-btn {
      padding: 6px 16px;
      border-radius: 8px;
      border: 2px solid var(--border-color, rgba(255, 255, 255, 0.15));
      background: var(--background, rgba(20, 20, 30, 0.8));
      color: var(--text-primary, white);
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: inherit;
    }

    .feature-flags-btn:hover {
      border-color: var(--primary-color, #00d9ff);
      color: var(--primary-color, #00d9ff);
      transform: scale(1.05);
    }

    .feature-flags-btn:active {
      transform: scale(0.98);
    }

    .icon {
      font-size: 12px;
    }
  `;

  private handleClick() {
    const modal = document.querySelector('feature-flags-modal') as any;
    if (modal) {
      modal.open();
    }
  }

  render() {
    return html`
      <button
        class="feature-flags-btn"
        @click=${this.handleClick}
        title="Open Feature Flags"
        data-testid="feature-flags-button"
      >
        <span>Feature Flags</span>
      </button>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'feature-flags-button': FeatureFlagsButton;
  }
}
