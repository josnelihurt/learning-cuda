import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import { ConnectionStatus, ConnectionInfo } from '../../domain/value-objects';
import { container } from '../../application/di';
import type { IWebSocketService } from '../../domain/interfaces/IWebSocketService';
import type { IWebRTCService } from '../../domain/interfaces/IWebRTCService';
import { grpcConnectionService } from '../../infrastructure/connection/grpc-connection-service';
import './connection-status-card';

const STATS_PANEL_EXPANDED_KEY = 'cuda-stats-panel-expanded';

export type CameraStatusType = 'success' | 'error' | 'warning' | 'inactive';

@customElement('stats-panel')
export class StatsPanel extends LitElement {
  @property({ type: String }) fps = '--';
  @property({ type: String }) time = '--ms';
  @property({ type: Number }) frames = 0;
  @property({ type: String }) cameraStatus = 'Inactive';
  @property({ type: String }) cameraStatusType: CameraStatusType = 'inactive';

  @state() private connections: ConnectionInfo[] = [];
  @state() private isExpanded: boolean = true;
  private updateInterval: number | null = null;

  private frameCount = 0;
  private fpsHistory: number[] = [];
  private processingTimes: number[] = [];
  private lastFrameTime = 0;

  static styles = css`
    :host {
      grid-area: stats;
      display: block;
      background: #2a2a2a;
      color: white;
      border-top: 2px solid #404040;
      padding: 2px var(--spacing-lg, 30px) 1px;
      z-index: var(--z-stats, 1000);
      position: relative;
      height: 35px;
      max-height: 35px;
      box-sizing: border-box;
      display: flex;
      align-items: center;
      transition: max-height 0.2s ease, padding 0.2s ease, border 0.2s ease, opacity 0.2s ease;
      overflow: hidden;
    }

    :host([collapsed]) {
      max-height: 0 !important;
      height: 0 !important;
      padding: 0 !important;
      border-top: none;
      min-height: 0;
      margin: 0;
      visibility: hidden;
    }

    :host([collapsed]) .stats-container {
      display: none;
    }

    :host([collapsed]) .toggle-button {
      visibility: visible !important;
      opacity: 1 !important;
    }

    .stats-container {
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--spacing-xl, 32px);
      padding: 0;
      overflow: hidden;
      width: 100%;
    }

    .stats-left {
      display: flex;
      align-items: center;
      gap: var(--spacing-lg, 24px);
      flex-shrink: 0;
      padding-left: var(--spacing-sm, 12px);
    }

    .connections-section {
      flex: 1;
      min-width: 0;
      overflow-x: auto;
      overflow-y: hidden;
      display: flex;
      justify-content: flex-end;
      align-items: flex-start;
      padding-right: 35px;
    }

    .connections-section::-webkit-scrollbar {
      height: 4px;
    }

    .connections-section::-webkit-scrollbar-track {
      background: rgba(255, 255, 255, 0.05);
    }

    .connections-section::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.2);
      border-radius: 2px;
    }

    .stat-item {
      display: flex;
      align-items: center;
      gap: var(--spacing-xs, 6px);
      font-size: 12px;
      color: #e0e0e0;
      white-space: nowrap;
      flex-shrink: 0;
    }

    .stat-label {
      color: #b0b0b0;
      font-weight: 500;
      font-size: 11px;
    }

    strong {
      font-weight: 600;
      font-size: 13px;
      min-width: 50px;
      text-align: left;
    }

    .camera-status {
      font-weight: 600;
      font-size: 12px;
    }

    .camera-status.success {
      color: #66ff66;
    }

    .camera-status.error {
      color: #ff6666;
    }

    .camera-status.warning {
      color: #ffaa00;
    }

    .camera-status.inactive {
      color: #b0b0b0;
    }

    .connections-grid {
      display: flex;
      gap: var(--spacing-xs, 6px);
      align-items: stretch;
      min-width: fit-content;
      justify-content: flex-end;
    }

    .toggle-button {
      position: fixed;
      right: var(--spacing-sm, 12px);
      bottom: 0;
      background: rgba(42, 42, 42, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-bottom: none;
      border-radius: 4px 4px 0 0;
      color: #e0e0e0;
      cursor: pointer;
      padding: 4px 6px;
      font-size: 12px;
      line-height: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      width: 20px;
      height: 20px;
      transition: background 0.2s ease, border-color 0.2s ease;
      z-index: 1001;
      flex-shrink: 0;
    }

    .toggle-button:hover {
      background: rgba(42, 42, 42, 1);
      border-color: rgba(255, 255, 255, 0.3);
    }

    .toggle-button:active {
      background: rgba(42, 42, 42, 0.8);
    }

    :host([collapsed]) .toggle-button {
      opacity: 1 !important;
      visibility: visible !important;
      display: flex !important;
    }

    @media (max-width: 1200px) {
      .stats-container {
        gap: var(--spacing-md, 16px);
      }
    }

    @media (max-width: 768px) {
      .stats-container {
        flex-direction: column;
        align-items: flex-start;
        gap: var(--spacing-sm, 12px);
        padding: var(--spacing-sm, 12px) 0;
        height: auto;
      }

      .stats-left {
        width: 100%;
        flex-wrap: wrap;
        gap: var(--spacing-sm, 12px);
      }

      .stat-item {
        gap: 4px;
        font-size: 12px;
      }

      .stat-label {
        display: none;
      }

      .connections-section {
        width: 100%;
      }

      .connections-grid {
        flex-direction: column;
        gap: var(--spacing-xs, 8px);
      }
    }
  `;

  connectedCallback(): void {
    super.connectedCallback();
    this.loadPanelState();
    this.updateConnections();
    this.updateInterval = window.setInterval(() => {
      this.updateConnections();
    }, 2000);
  }

  private loadPanelState(): void {
    try {
      const savedState = localStorage.getItem(STATS_PANEL_EXPANDED_KEY);
      if (savedState !== null) {
        const expanded = savedState === 'true';
        this.isExpanded = expanded;
        if (!expanded) {
          this.setAttribute('collapsed', '');
          document.documentElement.style.setProperty('--stats-height', '0px');
        } else {
          this.removeAttribute('collapsed');
          document.documentElement.style.setProperty('--stats-height', '35px');
        }
      } else {
        if (!this.hasAttribute('collapsed')) {
          document.documentElement.style.setProperty('--stats-height', '35px');
        }
      }
    } catch {
      // ignore storage errors
      if (!this.hasAttribute('collapsed')) {
        document.documentElement.style.setProperty('--stats-height', '35px');
      }
    }
  }

  private savePanelState(): void {
    try {
      localStorage.setItem(STATS_PANEL_EXPANDED_KEY, String(this.isExpanded));
    } catch {
      // ignore storage errors
    }
  }

  disconnectedCallback(): void {
    super.disconnectedCallback();
    if (this.updateInterval !== null) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }

  @state() private wsService: (IWebSocketService & { getConnectionStatus?: () => any }) | null = null;
  @state() private webrtcService: (IWebRTCService & { getConnectionStatus?: () => any }) | null = null;

  private updateConnections(): void {
    try {
      const wsStatus = this.wsService?.getConnectionStatus?.() || { state: 'disconnected' as const, lastRequest: null, lastRequestTime: null };
      const webrtcStatus = this.webrtcService?.getConnectionStatus?.() || { state: 'disconnected' as const, lastRequest: null, lastRequestTime: null };
      const grpcStatus = grpcConnectionService.getConnectionStatus();

      this.connections = [
        ConnectionInfo.websocket(ConnectionStatus.create(wsStatus.state, wsStatus.lastRequest, wsStatus.lastRequestTime)),
        ConnectionInfo.grpc(ConnectionStatus.create(grpcStatus.state, grpcStatus.lastRequest, grpcStatus.lastRequestTime)),
        ConnectionInfo.webrtc(ConnectionStatus.create(webrtcStatus.state, webrtcStatus.lastRequest, webrtcStatus.lastRequestTime)),
      ];
    } catch (error) {
      console.error('Error updating connections:', error);
    }
  }

  setWebSocketService(service: IWebSocketService & { getConnectionStatus?: () => any }): void {
    this.wsService = service;
    this.updateConnections();
  }

  setWebRTCService(service: IWebRTCService & { getConnectionStatus?: () => any }): void {
    this.webrtcService = service;
    this.updateConnections();
  }


  private togglePanel() {
    this.isExpanded = !this.isExpanded;
    if (this.isExpanded) {
      this.removeAttribute('collapsed');
      document.documentElement.style.setProperty('--stats-height', '35px');
    } else {
      this.setAttribute('collapsed', '');
      document.documentElement.style.setProperty('--stats-height', '0px');
    }
    this.savePanelState();
  }

  render() {
    return html`
      <div class="stats-container">
        <div class="stats-left">
          <div class="stat-item">
            <span class="stat-label">FPS:</span>
            <strong>${this.fps}</strong>
          </div>
          <div class="stat-item">
            <span class="stat-label">Time:</span>
            <strong>${this.time}</strong>
          </div>
          <div class="stat-item">
            <span class="stat-label">Frames:</span>
            <strong>${this.frames}</strong>
          </div>
          <div class="stat-item">
            <span class="camera-status ${this.cameraStatusType}">${this.cameraStatus}</span>
          </div>
        </div>
        <div class="connections-section">
          <div class="connections-grid">
            ${this.connections.map(
              (conn) => html`<connection-status-card .connection=${conn}></connection-status-card>`
            )}
          </div>
        </div>
      </div>
      <button 
        class="toggle-button" 
        @click=${this.togglePanel}
        title=${this.isExpanded ? 'Hide panel' : 'Show panel'}
      >
        ${this.isExpanded ? '▼' : '▲'}
      </button>
    `;
  }

  incrementFrameCount(): void {
    this.frameCount++;
    this.frames = this.frameCount;
  }

  updateProcessingStats(processingTime: number): void {
    this.incrementFrameCount();

    const instantFPS = 1000 / processingTime;
    this.fpsHistory.push(instantFPS);
    if (this.fpsHistory.length > 10) this.fpsHistory.shift();

    const avgFPS = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
    this.fps = avgFPS.toFixed(1);

    this.processingTimes.push(processingTime);
    if (this.processingTimes.length > 10) this.processingTimes.shift();

    const avgTime = this.processingTimes.reduce((a, b) => a + b, 0) / this.processingTimes.length;
    this.time = avgTime.toFixed(0) + 'ms';
  }

  updateCameraStatus(status: string, type: CameraStatusType): void {
    this.cameraStatus = status;
    this.cameraStatusType = type;
  }

  updateWebSocketStatus(status: 'connected' | 'disconnected' | 'connecting', text: string): void {
    this.updateConnections();
  }

  reset(): void {
    this.frameCount = 0;
    this.fpsHistory = [];
    this.processingTimes = [];
    this.lastFrameTime = 0;

    this.fps = '--';
    this.time = '--ms';
    this.frames = 0;
    this.updateCameraStatus('Inactive', 'inactive');
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'stats-panel': StatsPanel;
  }
}
