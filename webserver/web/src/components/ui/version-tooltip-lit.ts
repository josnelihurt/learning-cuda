import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { systemInfoService } from '../../services/system-info-service';

@customElement('version-tooltip-lit')
export class VersionTooltipLit extends LitElement {
  @state() private isOpen = false;
  @state() private cppVersion = 'Loading...';
  @state() private goVersion = 'Loading...';
  @state() private jsVersion = 'Loading...';
  @state() private branch = 'Loading...';
  @state() private buildTime = 'Loading...';
  @state() private commitHash = 'Loading...';
  @state() private isLoading = true;

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
      min-width: 50px;
    }

    .version-value {
      color: #666;
      font-family: monospace;
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    console.log('VersionTooltipLit connected');
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

  private async loadSystemInfo() {
    try {
      const systemInfo = await systemInfoService.getSystemInfo();
      
      this.cppVersion = systemInfo.version?.cppVersion || 'Unknown';
      this.goVersion = systemInfo.version?.goVersion || 'Unknown';
      this.jsVersion = systemInfo.version?.jsVersion || 'Unknown';
      this.branch = systemInfo.version?.branch || 'Unknown';
      this.buildTime = systemInfo.version?.buildTime || 'Unknown';
      this.commitHash = systemInfo.version?.commitHash || 'Unknown';
      this.isLoading = false;
    } catch (error) {
      console.error('Failed to load system info:', error);
      this.cppVersion = 'Error';
      this.goVersion = 'Error';
      this.jsVersion = 'Error';
      this.branch = 'Error';
      this.buildTime = 'Error';
      this.commitHash = 'Error';
      this.isLoading = false;
    }
  }

  private handleSlotClick() {
    console.log('Slot clicked');
    this.isOpen = !this.isOpen;
  }

  private handleCloseClick(e: Event) {
    e.stopPropagation();
    console.log('Close clicked');
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
          <div class="version-item">
            <span class="version-label">C++:</span>
            <span class="version-value">${this.cppVersion}</span>
          </div>
          <div class="version-item">
            <span class="version-label">Go:</span>
            <span class="version-value">${this.goVersion}</span>
          </div>
          <div class="version-item">
            <span class="version-label">JS:</span>
            <span class="version-value">${this.jsVersion}</span>
          </div>
          <div class="version-item">
            <span class="version-label">Branch:</span>
            <span class="version-value">${this.branch}</span>
          </div>
          <div class="version-item">
            <span class="version-label">Build:</span>
            <span class="version-value">${this.buildTime}</span>
          </div>
          <div class="version-item">
            <span class="version-label">Commit:</span>
            <span class="version-value">${this.commitHash}</span>
          </div>
          <div class="version-item">
            <span class="version-label">Dockerfiles:</span>
            <span class="version-value"></span>
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
