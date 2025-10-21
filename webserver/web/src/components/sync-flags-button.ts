import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { createPromiseClient, PromiseClient, Interceptor, ConnectError } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService } from '../gen/image_processing_connect';
import { telemetryService } from '../services/telemetry-service';

const tracingInterceptor: Interceptor = (next) => async (req) => {
    const headers = telemetryService.getTraceHeaders();
    for (const [key, value] of Object.entries(headers)) {
        req.header.set(key, value);
    }
    return await next(req);
};

@customElement('sync-flags-button')
export class SyncFlagsButton extends LitElement {
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
            display: inline-block;
        }

        .sync-btn {
            padding: 6px 16px;
            border-radius: 8px;
            border: 2px solid var(--border-color, #e5e7eb);
            background: var(--background, #ffffff);
            color: var(--text-primary, #1f2937);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: inherit;
        }

        .sync-btn:hover:not(:disabled) {
            border-color: var(--primary-color, #3b82f6);
            background: var(--primary-color, #3b82f6);
            color: white;
            transform: scale(1.05);
        }

        .sync-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .sync-btn.syncing {
            background: var(--primary-light, #60a5fa);
            color: white;
            border-color: var(--primary-light, #60a5fa);
        }
    `;

    private async handleSync() {
        if (this.syncing) return;

        this.syncing = true;

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
                            'message': response.message,
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

    render() {
        return html`
            <button 
                class="sync-btn ${this.syncing ? 'syncing' : ''}"
                @click=${this.handleSync}
                ?disabled=${this.syncing}
                title="Sync feature flags to Flipt"
            >
                <span>${this.syncing ? 'Syncing...' : 'Sync Flags'}</span>
            </button>
        `;
    }
}

declare global {
    interface HTMLElementTagNameMap {
        'sync-flags-button': SyncFlagsButton;
    }
}

