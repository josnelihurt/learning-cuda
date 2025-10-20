import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';

declare const __APP_VERSION__: string;
declare const __APP_BRANCH__: string;
declare const __BUILD_TIME__: string;

@customElement('version-footer')
export class VersionFooter extends LitElement {
    @state() private cppVersion = '...';
    @state() private goVersion = '...';
    @state() private jsVersion = __APP_VERSION__;
    @state() private branch = __APP_BRANCH__;
    @state() private buildTime = __BUILD_TIME__;

    static styles = css`
        :host {
            display: block;
            padding: 8px 16px;
            font-size: 11px;
            color: #666;
            font-family: 'Courier New', monospace;
            background: rgba(0, 0, 0, 0.02);
        }

        .versions {
            display: flex;
            gap: 16px;
            align-items: center;
        }

        .version {
            display: flex;
            gap: 4px;
        }

        .label {
            font-weight: 600;
        }

        .value {
            font-family: monospace;
        }
    `;

    async connectedCallback() {
        super.connectedCallback();
        await this.loadVersions();
    }

    async loadVersions() {
        try {
            const response = await fetch('/api/processor/capabilities');
            const data = await response.json();
            if (data.capabilities) {
                this.cppVersion = data.capabilities.libraryVersion || '?';
                this.goVersion = data.capabilities.apiVersion || '?';
            }
        } catch (e) {
            console.warn('Failed to load backend versions', e);
        }
    }

    render() {
        return html`
            <div class="versions">
                <div class="version">
                    <span class="label">C++:</span>
                    <span class="value">${this.cppVersion}</span>
                </div>
                <div class="version">
                    <span class="label">Go:</span>
                    <span class="value">${this.goVersion}</span>
                </div>
                <div class="version">
                    <span class="label">JS:</span>
                    <span class="value">${this.jsVersion}</span>
                </div>
                <div class="version">
                    <span class="label">Branch:</span>
                    <span class="value">${this.branch}</span>
                </div>
                <div class="version">
                    <span class="label">Build:</span>
                    <span class="value">${new Date(this.buildTime).toLocaleString()}</span>
                </div>
            </div>
        `;
    }
}

declare global {
    interface HTMLElementTagNameMap {
        'version-footer': VersionFooter;
    }
}


