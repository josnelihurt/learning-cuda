import { LitElement, html, css } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { ConnectionInfo } from '../../domain/value-objects';

@customElement('connection-status-card')
export class ConnectionStatusCard extends LitElement {
  @property({ type: Object }) connection!: ConnectionInfo;

  static styles = css`
    :host {
      display: block;
    }

    .connection-item {
      display: flex;
      flex-direction: row;
      align-items: flex-start;
      gap: var(--spacing-sm, 8px);
      padding: 3px var(--spacing-sm, 10px);
      background: rgba(255, 255, 255, 0.05);
      border-radius: 6px;
      font-size: 10px;
      width: 180px;
      height: 32px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      flex-shrink: 0;
      box-sizing: border-box;
      overflow: hidden;
    }

    .connection-left {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 3px;
      min-width: 45px;
      flex-shrink: 0;
      height: 100%;
    }

    .connection-header {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
      font-weight: 600;
      font-size: 11px;
      flex-shrink: 0;
      position: relative;
      cursor: help;
      width: 100%;
    }

    .connection-tooltip {
      padding: 6px 10px;
      background: rgba(0, 0, 0, 0.98);
      color: white;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 400;
      white-space: nowrap;
      z-index: 999999;
      pointer-events: none;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(4px);
    }

    .connection-protocol-name {
      font-size: 9px;
      font-weight: 600;
      color: #e0e0e0;
      text-align: center;
    }

    .connection-indicator {
      display: inline-block;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .connection-indicator.connected {
      background: #66ff66;
      box-shadow: 0 0 6px #66ff66;
    }

    .connection-indicator.disconnected {
      background: #ff6666;
      box-shadow: 0 0 6px #ff6666;
    }

    .connection-indicator.connecting {
      background: #ffaa00;
      box-shadow: 0 0 6px #ffaa00;
      animation: pulse 1.5s ease-in-out infinite;
    }

    .connection-indicator.error {
      background: #ff6666;
      box-shadow: 0 0 6px #ff6666;
    }

    .connection-right {
      display: flex;
      flex-direction: column;
      gap: 2px;
      flex: 1;
      min-width: 0;
      justify-content: center;
      height: 100%;
    }

    .connection-detail {
      font-size: 9px;
      color: #e0e0e0;
      line-height: 1.1;
      display: flex;
      align-items: center;
      gap: 3px;
      white-space: nowrap;
      flex-shrink: 0;
      width: 100%;
    }

    .connection-detail-label {
      color: #888;
      font-weight: 500;
      font-size: 8px;
      flex-shrink: 0;
    }

    .connection-detail-value {
      color: #e0e0e0;
      font-weight: 400;
      font-size: 9px;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1;
    }

    .connection-time {
      font-size: 8px;
      color: #b0b0b0;
      text-align: left;
      width: 100%;
      line-height: 1.1;
    }

    @keyframes pulse {
      0%,
      100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }

    @media (max-width: 1200px) {
      .connection-item {
        padding: var(--spacing-xs, 8px) var(--spacing-sm, 12px);
        gap: var(--spacing-sm, 12px);
      }

      .connection-detail {
        font-size: 11px;
      }
    }

    @media (max-width: 768px) {
      .connection-item {
        flex-direction: column;
        align-items: flex-start;
        min-width: auto;
        max-width: 100%;
        width: 100%;
        padding: var(--spacing-xs, 8px) var(--spacing-sm, 12px);
        gap: var(--spacing-xs, 6px);
      }

      .connection-header {
        min-width: auto;
        width: 100%;
      }
    }
  `;

  private truncateRequestName(request: string, maxLength: number = 12): string {
    if (!request || request === 'N/A') {
      return request;
    }
    if (request.length <= maxLength) {
      return request;
    }
    return request.substring(0, maxLength) + '...';
  }

  private handleHeaderMouseEnter(event: MouseEvent, state: string) {
    const target = event.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    const tooltip = document.createElement('div');
    tooltip.className = 'connection-tooltip';
    tooltip.textContent = state;
    tooltip.style.position = 'fixed';
    tooltip.style.left = `${rect.left + rect.width / 2}px`;
    tooltip.style.top = `${rect.top - 8}px`;
    tooltip.style.transform = 'translate(-50%, -100%)';
    tooltip.style.zIndex = '999999';
    tooltip.style.padding = '6px 10px';
    tooltip.style.background = 'rgba(0, 0, 0, 0.98)';
    tooltip.style.color = 'white';
    tooltip.style.borderRadius = '4px';
    tooltip.style.fontSize = '11px';
    tooltip.style.fontWeight = '400';
    tooltip.style.whiteSpace = 'nowrap';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.6)';
    tooltip.style.backdropFilter = 'blur(4px)';
    document.body.appendChild(tooltip);
    (target as any).__tooltip = tooltip;
  }

  private handleHeaderMouseLeave(event: MouseEvent) {
    const target = event.currentTarget as HTMLElement;
    const tooltip = (target as any).__tooltip;
    if (tooltip) {
      tooltip.remove();
      (target as any).__tooltip = null;
    }
  }

  private handleRequestMouseEnter(event: MouseEvent, fullRequest: string) {
    const target = event.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    const tooltip = document.createElement('div');
    tooltip.className = 'connection-tooltip';
    tooltip.textContent = fullRequest;
    tooltip.style.position = 'fixed';
    tooltip.style.left = `${rect.left + rect.width / 2}px`;
    tooltip.style.top = `${rect.top - 8}px`;
    tooltip.style.transform = 'translate(-50%, -100%)';
    tooltip.style.zIndex = '999999';
    tooltip.style.padding = '6px 10px';
    tooltip.style.background = 'rgba(0, 0, 0, 0.98)';
    tooltip.style.color = 'white';
    tooltip.style.borderRadius = '4px';
    tooltip.style.fontSize = '11px';
    tooltip.style.fontWeight = '400';
    tooltip.style.whiteSpace = 'nowrap';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.6)';
    tooltip.style.backdropFilter = 'blur(4px)';
    document.body.appendChild(tooltip);
    (target as any).__tooltip = tooltip;
  }

  private handleRequestMouseLeave(event: MouseEvent) {
    const target = event.currentTarget as HTMLElement;
    const tooltip = (target as any).__tooltip;
    if (tooltip) {
      tooltip.remove();
      (target as any).__tooltip = null;
    }
  }

  render() {
    if (!this.connection) {
      return html``;
    }

    const requestName = this.truncateRequestName(this.connection.status.getLastRequestDisplay(), 12);
    const stateDisplay = this.connection.getStateDisplay();

    return html`
      <div class="connection-item">
        <div class="connection-left">
          <div 
            class="connection-header" 
            data-state="${stateDisplay}"
            @mouseenter=${(e: MouseEvent) => this.handleHeaderMouseEnter(e, stateDisplay)}
            @mouseleave=${(e: MouseEvent) => this.handleHeaderMouseLeave(e)}
          >
            <span class="connection-protocol-name">${this.connection.label}</span>
            <span class="connection-indicator ${this.connection.status.state}"></span>
          </div>
        </div>
        <div class="connection-right">
          <div class="connection-detail">
            <span class="connection-detail-label">Req:</span>
            <span 
              class="connection-detail-value"
              @mouseenter=${(e: MouseEvent) => {
                const fullRequest = this.connection.status.getLastRequestDisplay();
                if (fullRequest && fullRequest !== 'N/A' && fullRequest.length > 12) {
                  this.handleRequestMouseEnter(e, fullRequest);
                }
              }}
              @mouseleave=${(e: MouseEvent) => this.handleRequestMouseLeave(e)}
            >${requestName}</span>
          </div>
          <div class="connection-time">${this.connection.status.getLastRequestTimeDisplay()}</div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'connection-status-card': ConnectionStatusCard;
  }
}

