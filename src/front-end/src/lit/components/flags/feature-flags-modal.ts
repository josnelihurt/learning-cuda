import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { createPromiseClient, PromiseClient, Interceptor } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService } from '../../../gen/config_service_connect';
import { telemetryService } from '../../../infrastructure/observability/telemetry-service';

const tracingInterceptor: Interceptor = (next) => async (req) => {
  const headers = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(headers)) {
    req.header.set(key, value);
  }
  return await next(req);
};

@customElement('feature-flags-modal')
export class FeatureFlagsModal extends LitElement {
  @state() private isOpen = false;
  @state() private loading = false;
  @state() private savingKey: string | null = null;
  @state() private flags: ManagedFlag[] = [];
  private client: PromiseClient<typeof ConfigService>;

  constructor() {
    super();
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      interceptors: [tracingInterceptor],
    });
    this.client = createPromiseClient(ConfigService, transport);
  }

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
      background: rgba(0, 0, 0, 0.6);
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
      width: 90vw;
      height: 85vh;
      max-width: 1198px;
      max-height: 800px;
      background: rgba(20, 20, 28, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
      backdrop-filter: blur(10px);
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
      padding: 12px 20px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      background: rgba(255, 255, 255, 0.02);
      backdrop-filter: blur(10px);
    }

    .modal-title {
      font-size: 16px;
      font-weight: 500;
      color: rgba(255, 255, 255, 0.9);
      margin: 0;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .close-btn {
      background: none;
      border: none;
      font-size: 18px;
      cursor: pointer;
      color: rgba(255, 255, 255, 0.5);
      padding: 2px;
      line-height: 1;
      transition: color 0.2s;
      width: 28px;
      height: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 4px;
    }

    .close-btn:hover {
      color: rgba(255, 255, 255, 0.8);
      background: rgba(255, 255, 255, 0.05);
    }

    .modal-content {
      flex: 1;
      overflow: auto;
      padding: 16px;
      background: #161821;
    }

    .table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    .cell {
      padding: 8px;
      text-align: left;
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    document.addEventListener('keydown', this.handleKeyDown);
    document.addEventListener('open-feature-flags-modal', this.open);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    document.removeEventListener('keydown', this.handleKeyDown);
    document.removeEventListener('open-feature-flags-modal', this.open);
  }

  private handleKeyDown = (e: KeyboardEvent) => {
    if (this.isOpen && e.key === 'Escape') {
      this.close();
    }
  };

  private handleBackdropClick = (e: MouseEvent) => {
    // Only close if clicking on the backdrop itself, not children
    if (e.target === e.currentTarget) {
      this.close();
    }
  };

  async open() {
    this.isOpen = true;
    this.setAttribute('open', '');
    await this.loadFlags();
    requestAnimationFrame(() => {
      const backdrop = this.shadowRoot?.querySelector('.backdrop');
      const modal = this.shadowRoot?.querySelector('.modal');
      if (backdrop) backdrop.classList.add('show');
      if (modal) modal.classList.add('show');
    });
  }

  close() {
    const backdrop = this.shadowRoot?.querySelector('.backdrop');
    const modal = this.shadowRoot?.querySelector('.modal');
    if (backdrop) backdrop.classList.remove('show');
    if (modal) modal.classList.remove('show');
    
    setTimeout(() => {
      this.isOpen = false;
      this.removeAttribute('open');
    }, 300);
  }

  private async loadFlags(): Promise<void> {
    this.loading = true;
    try {
      const response = await this.client.listFeatureFlags({});
      this.flags = response.flags.map((flag) => ({
        key: flag.key,
        name: flag.name,
        type: flag.type,
        enabled: flag.enabled,
        defaultValue: flag.defaultValue,
        description: flag.description,
      }));
    } catch (error) {
      console.error('Feature flags load failed', error);
    } finally {
      this.loading = false;
    }
  }

  private updateFlagField(
    key: string,
    field: 'enabled' | 'defaultValue',
    value: boolean | string
  ): void {
    this.flags = this.flags.map((item) => {
      if (item.key !== key) {
        return item;
      }
      if (field === 'enabled') {
        return { ...item, enabled: Boolean(value) };
      }
      return { ...item, defaultValue: String(value) };
    });
  }

  private async saveFlag(flag: ManagedFlag): Promise<void> {
    this.savingKey = flag.key;
    try {
      await this.client.upsertFeatureFlag({
        flag: {
          key: flag.key,
          name: flag.name,
          type: flag.type,
          enabled: flag.enabled,
          defaultValue: flag.defaultValue,
          description: flag.description,
        },
      });
      await this.loadFlags();
    } catch (error) {
      console.error('Feature flag update failed', error);
    } finally {
      this.savingKey = null;
    }
  }

  render() {
    return html`
      <div class="backdrop ${this.isOpen ? 'show' : ''}" @click=${this.handleBackdropClick}></div>
      <div class="modal ${this.isOpen ? 'show' : ''}">
        <div class="modal-header">
          <h2 class="modal-title">Feature Flags</h2>
          <div class="header-actions">
            <button class="close-btn" @click=${this.close} title="Close">×</button>
          </div>
        </div>
        <div class="modal-content">
          ${this.loading
            ? html`<div>Loading flags...</div>`
            : html`<table class="table">
                <thead>
                  <tr>
                    <th class="cell">Key</th>
                    <th class="cell">Type</th>
                    <th class="cell">Enabled</th>
                    <th class="cell">Default</th>
                    <th class="cell">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  ${this.flags.map(
                    (flag) => html`<tr>
                      <td class="cell">${flag.key}</td>
                      <td class="cell">${flag.type}</td>
                      <td class="cell">
                        <input
                          type="checkbox"
                          .checked=${flag.enabled}
                          @change=${(event: Event) =>
                            this.updateFlagField(
                              flag.key,
                              'enabled',
                              (event.target as HTMLInputElement).checked
                            )}
                        />
                      </td>
                      <td class="cell">
                        <input
                          .value=${flag.defaultValue}
                          @input=${(event: Event) =>
                            this.updateFlagField(
                              flag.key,
                              'defaultValue',
                              (event.target as HTMLInputElement).value
                            )}
                        />
                      </td>
                      <td class="cell">
                        <button
                          type="button"
                          class="close-btn"
                          ?disabled=${this.savingKey === flag.key}
                          @click=${() => this.saveFlag(flag)}
                        >
                          ${this.savingKey === flag.key ? 'Saving...' : 'Save'}
                        </button>
                      </td>
                    </tr>`
                  )}
                </tbody>
              </table>`}
        </div>
      </div>
    `;
  }
}

type ManagedFlag = {
  key: string;
  name: string;
  type: string;
  enabled: boolean;
  defaultValue: string;
  description: string;
};

declare global {
  interface HTMLElementTagNameMap {
    'feature-flags-modal': FeatureFlagsModal;
  }
}
