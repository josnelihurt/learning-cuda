import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { createPromiseClient } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService } from '../../gen/config_service_connect';
import { GetSystemInfoRequest, GetSystemInfoResponse } from '../../gen/config_service_pb';

interface VersionField {
  label: string;
  value: string;
}

@customElement('version-footer')
export class VersionFooter extends LitElement {
  @state() private versionFields: VersionField[] = [];

  static styles = css`
    :host {
      display: block;
      padding: 8px 16px;
      font-size: 11px;
      color: #666;
      font-family: 'Courier New', monospace;
      background: rgba(0, 0, 0, 0.04);
    }

    .versions {
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
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

  private extractVersionFields(systemInfo: GetSystemInfoResponse): VersionField[] {
    const fields: VersionField[] = [];
    
    if (!systemInfo.version) {
      return fields;
    }

    const version = systemInfo.version;
    const fieldMap: Array<{ key: string; label: string; format?: (v: string) => string }> = [
      { key: 'goVersion', label: 'Go' },
      { key: 'cppVersion', label: 'C++' },
      { key: 'protoVersion', label: 'Proto' },
      { key: 'branch', label: 'Branch' },
      { key: 'buildTime', label: 'Build', format: (v) => new Date(v).toLocaleString() },
      { key: 'commitHash', label: 'Commit' },
    ];

    for (const { key, label, format } of fieldMap) {
      const value = (version as any)[key] as string | undefined;
      if (value && typeof value === 'string' && value.trim() !== '') {
        fields.push({
          label,
          value: format ? format(value) : value,
        });
      }
    }

    return fields;
  }

  async loadVersions() {
    try {
      const transport = createConnectTransport({
        baseUrl: window.location.origin,
        useHttpGet: true,
      });
      const client = createPromiseClient(ConfigService, transport);
      const systemInfo = await client.getSystemInfo(new GetSystemInfoRequest({}));
      this.versionFields = this.extractVersionFields(systemInfo);
    } catch (e) {
      console.warn('Failed to load backend versions', e);
    }
  }

  render() {
    return html`
      <div class="versions">
        ${this.versionFields.map(
          (field) => html`
            <div class="version">
              <span class="label">${field.label}:</span>
              <span class="value">${field.value}</span>
            </div>
          `
        )}
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'version-footer': VersionFooter;
  }
}
