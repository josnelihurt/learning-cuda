import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { systemInfoService } from '../../infrastructure/external/system-info-service';
import { grpcVersionService } from '../../infrastructure/external/grpc-version-service';
import { GetSystemInfoResponse } from '../../gen/config_service_pb';
import { GetVersionInfoResponse } from '../../gen/image_processor_service_pb';

interface VersionField {
  label: string;
  value: string;
  path: string;
}

@customElement('version-tooltip-lit')
export class VersionTooltipLit extends LitElement {
  @state() private isOpen = false;
  @state() private versionFields: VersionField[] = [];
  @state() private environment = 'Loading...';
  @state() private isLoading = true;
  @state() private grpcVersionInfo: GetVersionInfoResponse | null = null;
  @state() private isLoadingGrpcVersion = false;

  static styles = css`
    :host {
      position: relative;
      display: inline-block;
    }

    .tooltip {
      position: absolute;
      top: calc(100% + 10px);
      right: 0;
      background: white;
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      max-width: 350px;
      z-index: 1000;
      display: none;
    }

    .tooltip.open {
      display: block;
    }

    .tooltip-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #eee;
    }

    .tooltip-title {
      font-weight: 600;
      color: #333;
      font-size: 14px;
    }

    .tooltip-close {
      background: none;
      border: none;
      color: #666;
      font-size: 18px;
      cursor: pointer;
      padding: 2px 6px;
      border-radius: 3px;
      transition: all 0.2s;
      line-height: 1;
    }

    .tooltip-close:hover {
      background: #f0f0f0;
      color: #333;
    }

    .version-info {
      font-family: 'Courier New', monospace;
      font-size: 12px;
    }

    .version-item {
      display: flex;
      gap: 8px;
      margin-bottom: 4px;
      align-items: center;
    }

    .version-label {
      font-weight: 600;
      color: #333;
      min-width: 120px;
    }

    .version-value {
      color: #666;
      font-family: monospace;
    }

    .environment-section {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #eee;
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    this.setupEventListeners();
    this.loadSystemInfo();
  }

  private setupEventListeners() {
    document.addEventListener('click', (e) => {
      if (this.isOpen) {
        const path = e.composedPath();
        const clickedInside = path.some(el => el === this || (el as HTMLElement).shadowRoot === this.shadowRoot);
        if (!clickedInside) {
          this.close();
        }
      }
    });
  }

  private extractVersionFields(systemInfo: GetSystemInfoResponse): VersionField[] {
    const fields: VersionField[] = [];
    
    if (!systemInfo.version) {
      return fields;
    }

    const version = systemInfo.version;
    const fieldMap: Array<{ key: string; label: string; format?: (v: string) => string }> = [
      { key: 'goVersion', label: 'Go Version' },
      { key: 'cppVersion', label: 'C++ Version' },
      { key: 'protoVersion', label: 'Proto Version' },
      { key: 'branch', label: 'Branch' },
      { key: 'buildTime', label: 'Build Time', format: (v) => new Date(v).toLocaleString() },
      { key: 'commitHash', label: 'Commit Hash' },
    ];

    for (const { key, label, format } of fieldMap) {
      const value = (version as any)[key] as string | undefined;
      if (value && typeof value === 'string' && value.trim() !== '') {
        fields.push({
          label,
          value: format ? format(value) : value,
          path: `version.${key}`,
        });
      }
    }

    return fields;
  }

  private async loadSystemInfo() {
    try {
      const systemInfo = await systemInfoService.getSystemInfo();
      
      this.versionFields = this.extractVersionFields(systemInfo);
      this.environment = systemInfo.environment || 'Unknown';
      this.isLoading = false;
    } catch (error) {
      console.error('Failed to load system info:', error);
      this.versionFields = [];
      this.environment = 'Error';
      this.isLoading = false;
    }
  }

  private async handleSlotClick() {
    this.isOpen = !this.isOpen;
    if (this.isOpen && !this.grpcVersionInfo && !this.isLoadingGrpcVersion) {
      await this.loadGrpcVersionInfo();
    }
  }

  private async loadGrpcVersionInfo() {
    this.isLoadingGrpcVersion = true;
    try {
      this.grpcVersionInfo = await grpcVersionService.getVersionInfo();
    } catch (error) {
      console.error('Failed to load gRPC version info:', error);
      this.grpcVersionInfo = null;
    } finally {
      this.isLoadingGrpcVersion = false;
    }
  }

  private handleCloseClick(e: Event) {
    e.stopPropagation();
    this.close();
  }

  private close() {
    this.isOpen = false;
  }

  render() {
    return html`
      <slot @click=${this.handleSlotClick}></slot>
      <div class="tooltip ${this.isOpen ? 'open' : ''}">
        <div class="tooltip-header">
          <span class="tooltip-title">Version Information</span>
          <button class="tooltip-close" @click=${this.handleCloseClick} title="Close">×</button>
        </div>
        <div class="version-info">
          ${this.isLoading
            ? html`<div class="version-item">Loading...</div>`
            : this.versionFields.map(
                (field) => html`
                  <div class="version-item">
                    <span class="version-label">${field.label}:</span>
                    <span class="version-value">${field.value}</span>
                  </div>
                `
              )
          }
          ${this.isOpen ? html`
            ${this.isLoadingGrpcVersion
              ? html`<div class="version-item"><span class="version-label">gRPC Server Version:</span><span class="version-value">Loading...</span></div>`
              : this.grpcVersionInfo
                ? html`
                    <div class="version-item">
                      <span class="version-label">gRPC Server Version:</span>
                      <span class="version-value">${this.grpcVersionInfo.serverVersion || 'unknown'}</span>
                    </div>
                    <div class="version-item">
                      <span class="version-label">C++ Library Version:</span>
                      <span class="version-value">${this.grpcVersionInfo.libraryVersion || 'unknown'}</span>
                    </div>
                  `
                : html`<div class="version-item"><span class="version-label">gRPC Server Version:</span><span class="version-value">Error loading</span></div>`
            }
          ` : ''}
          <div class="environment-section">
            <div class="version-item">
              <span class="version-label">Environment:</span>
              <span class="version-value">${this.environment}</span>
            </div>
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'version-tooltip-lit': VersionTooltipLit;
  }
}
