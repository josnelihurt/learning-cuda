import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';

@customElement('flipt-proxy')
export class FliptProxy extends LitElement {
  @state() private fliptUrl = '';

  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
    }

    .proxy-container {
      width: 100%;
      height: 100%;
      border: none;
      background: white;
    }

    .loading {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: #666;
      font-size: 16px;
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    this.loadFliptUrl();
  }

  private async loadFliptUrl() {
    try {
      // Get Flipt URL from tools service
      const response = await fetch('/cuda_learning.ConfigService/GetAvailableTools', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      if (response.ok) {
        const data = await response.json();
        for (const category of data.categories) {
          for (const tool of category.tools) {
            if (tool.name.toLowerCase().includes('flipt') && tool.url) {
              this.fliptUrl = tool.url;
              break;
            }
          }
        }
      }
    } catch (error) {
      console.error('Failed to load Flipt URL:', error);
      // Fallback URL
      this.fliptUrl = `http://${window.location.hostname}:8081`;
    }
  }

  render() {
    if (!this.fliptUrl) {
      return html`<div class="loading">Loading Flipt...</div>`;
    }

    return html`
      <iframe
        class="proxy-container"
        src="${this.fliptUrl}"
        title="Flipt Feature Flags"
        sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-top-navigation allow-modals"
        allow="fullscreen"
        referrerpolicy="no-referrer-when-downgrade"
        loading="lazy"
      ></iframe>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'flipt-proxy': FliptProxy;
  }
}
