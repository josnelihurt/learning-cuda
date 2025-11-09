import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { createPromiseClient, PromiseClient, Interceptor, ConnectError } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService } from '../../gen/config_service_connect';
import { telemetryService } from '../../services/telemetry-service';
import { logger } from '../../services/otel-logger';
import { toolsService } from '../../services/tools-service';

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
  @state() private syncing = false;
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

    .sync-btn {
      padding: 6px 12px;
      border-radius: 4px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      background: rgba(255, 255, 255, 0.05);
      color: rgba(255, 255, 255, 0.8);
      font-size: 13px;
      font-weight: 400;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .sync-btn:hover:not(:disabled) {
      background: rgba(0, 217, 255, 0.15);
      border-color: rgba(0, 217, 255, 0.4);
      color: #00d9ff;
      transform: scale(1.01);
    }

    .sync-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .sync-btn.syncing {
      background: rgba(0, 217, 255, 0.2);
      border-color: rgba(0, 217, 255, 0.5);
      color: #00d9ff;
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
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .iframe-container {
      flex: 1;
      position: relative;
      overflow: hidden;
      border-radius: 0 0 8px 8px;
    }

    .iframe-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 1;
      pointer-events: none;
    }

    .iframe {
      width: 100%;
      height: 100%;
      border: none;
      background: white;
      pointer-events: auto;
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
    // Ensure tools service is initialized to get Flipt URL
    if (!toolsService.isInitialized()) {
      await toolsService.initialize();
    }
    
    this.isOpen = true;
    this.setAttribute('open', '');
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

  private async handleSync() {
    if (this.syncing) return;

    this.syncing = true;
    this.requestUpdate();

    try {
      await telemetryService.withSpanAsync(
        'flipt.sync_flags',
        {
          'flipt.operation': 'sync',
          'rpc.service': 'ConfigService',
          'rpc.method': 'SyncFeatureFlags',
        },
        async (span) => {
          const startTime = performance.now();

          try {
            span.addEvent('Calling syncFeatureFlags RPC');

            const response = await this.client.syncFeatureFlags({});

            const endTime = performance.now();
            const duration = endTime - startTime;

            span.setAttribute('rpc.response_time_ms', duration);
            span.setAttribute('flipt.sync_success', true);
            span.setAttribute('flipt.message', response.message);

            this.showToast('success', response.message);
            logger.info('Flags synced successfully', {
              message: response.message,
            });

            span.addEvent('Sync completed successfully');
          } catch (error) {
            span.setAttribute('error', true);

            if (error instanceof ConnectError) {
              span.setAttribute('error.code', error.code);
              span.setAttribute('error.message', error.message);
              span.recordException(error);

              logger.error('Failed to sync flags ConnectError', {
                'error.code': String(error.code),
                'error.message': error.message,
                'error.details': error.rawMessage,
              });

              this.showToast('error', `Sync failed: ${error.message}`);
            } else {
              const errorMsg = error instanceof Error ? error.message : String(error);
              span.setAttribute('error.message', errorMsg);
              span.recordException(error as Error);

              logger.error('Failed to sync flags', {
                'error.message': error instanceof Error ? error.message : String(error),
              });
              this.showToast('error', 'Failed to connect to server');
            }

            throw error;
          }
        }
      );
    } finally {
      this.syncing = false;
      this.requestUpdate();
    }
  }

  private showToast(type: 'success' | 'error', message: string): void {
    const toastContainer = document.querySelector('toast-container');
    if (toastContainer) {
      if (type === 'success' && 'success' in toastContainer) {
        (toastContainer as any).success('Feature Flags', message);
      } else if (type === 'error' && 'error' in toastContainer) {
        (toastContainer as any).error('Sync Error', message);
      }
    }
  }

  private getFliptUrl(): string {
    // Use the proxy route to bypass CSP restrictions
    return `${window.location.origin}/flipt/#/namespaces/default/flags`;
  }

  render() {
    return html`
      <div class="backdrop ${this.isOpen ? 'show' : ''}" @click=${this.handleBackdropClick}></div>
      <div class="modal ${this.isOpen ? 'show' : ''}">
        <div class="modal-header">
          <h2 class="modal-title">Feature Flags</h2>
          <div class="header-actions">
            <button
              class="sync-btn ${this.syncing ? 'syncing' : ''}"
              @click=${this.handleSync}
              ?disabled=${this.syncing}
              title="Sync feature flags to Flipt"
            >
              <span>${this.syncing ? 'Syncing...' : 'Sync'}</span>
            </button>
            <button class="close-btn" @click=${this.close} title="Close">×</button>
          </div>
        </div>
        <div class="modal-content">
          <div class="iframe-container">
            <div class="iframe-overlay" @click=${this.handleBackdropClick}></div>
            <iframe
              class="iframe"
              src=${this.getFliptUrl()}
              title="Flipt Feature Flags"
              sandbox="allow-same-origin allow-scripts allow-forms allow-popups allow-top-navigation"
              allow="fullscreen"
              referrerpolicy="no-referrer-when-downgrade"
            ></iframe>
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'feature-flags-modal': FeatureFlagsModal;
  }
}
