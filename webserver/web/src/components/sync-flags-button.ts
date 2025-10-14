import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { telemetryService } from '../services/telemetry-service';

interface SyncFlagsResponse {
  success: boolean;
  message: string;
  flags?: Record<string, boolean>;
  errors?: string[];
}

@customElement('sync-flags-button')
export class SyncFlagsButton extends LitElement {
    @state() private syncing = false;

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
                    'http.method': 'POST',
                    'http.url': '/api/flipt/sync',
                },
                async (span) => {
                    const startTime = performance.now();

                    try {
                        const traceHeaders = telemetryService.getTraceHeaders();
                        
                        const response = await fetch('/api/flipt/sync', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                ...traceHeaders,
                            },
                        });

                        const endTime = performance.now();
                        const duration = endTime - startTime;

                        span.setAttribute('http.status_code', response.status);
                        span.setAttribute('http.response_time_ms', duration);

                        const data: SyncFlagsResponse = await response.json();

                        span.setAttribute('flipt.sync_success', data.success);
                        
                        if (data.flags) {
                            span.setAttribute('flipt.flags_count', Object.keys(data.flags).length);
                            span.setAttribute('flipt.flags', JSON.stringify(data.flags));
                        }

                        if (data.success) {
                            this.showToast('success', data.message);
                            if (data.flags) {
                                console.log('✓ Synced flags to Flipt:', data.flags);
                            }
                        } else {
                            this.showToast('error', data.message);
                            if (data.errors) {
                                span.setAttribute('flipt.errors', JSON.stringify(data.errors));
                                console.error('Sync errors:', data.errors);
                            }
                        }
                    } catch (error) {
                        span.setAttribute('error', true);
                        span.setAttribute('error.message', error instanceof Error ? error.message : String(error));
                        console.error('Failed to sync flags:', error);
                        this.showToast('error', 'Failed to connect to server');
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
        if (toastContainer && 'addToast' in toastContainer) {
            (toastContainer as any).addToast(message, type);
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

