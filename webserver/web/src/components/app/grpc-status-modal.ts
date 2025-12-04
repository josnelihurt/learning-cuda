import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { createPromiseClient, PromiseClient, Interceptor, ConnectError } from '@connectrpc/connect';
import { createConnectTransport } from '@connectrpc/connect-web';
import { ConfigService } from '../../gen/config_service_connect';
import { GetProcessorStatusResponse } from '../../gen/config_service_pb';
import { RemoteManagementService as RemoteManagementServiceClient } from '../../gen/remote_management_service_connect';
import { 
  CheckAcceleratorHealthResponse, 
  AcceleratorHealthStatus,
  StartJetsonNanoStatus
} from '../../gen/remote_management_service_pb';
import { telemetryService } from '../../infrastructure/observability/telemetry-service';
import { logger } from '../../infrastructure/observability/otel-logger';
import { processorCapabilitiesService } from '../../services/processor-capabilities-service';
import { remoteManagementService, StartJetsonNanoEvent } from '../../infrastructure/external/remote-management-service';

const tracingInterceptor: Interceptor = (next) => async (req) => {
  const headers = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(headers)) {
    req.header.set(key, value);
  }
  return await next(req);
};

@customElement('grpc-status-modal')
export class GrpcStatusModal extends LitElement {
  @state() private isOpen = false;
  @state() private isMinimized = false;
  @state() private status: GetProcessorStatusResponse | null = null;
  @state() private acceleratorHealth: CheckAcceleratorHealthResponse | null = null;
  @state() private isLoading = false;
  @state() private lastCheck: Date | null = null;
  @state() private error: string | null = null;
  @state() private terminalOutput: string[] = [];
  @state() private isStartingJetson = false;
  private client: PromiseClient<typeof ConfigService>;
  private refreshInterval: number | null = null;

  constructor() {
    super();
    const transport = createConnectTransport({
      baseUrl: window.location.origin,
      interceptors: [tracingInterceptor],
      useHttpGet: true,
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
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .status-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
    }

    .status-indicator.online {
      background: #22c55e;
      box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
    }

    .status-indicator.offline {
      background: #ef4444;
      box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
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
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .info-section {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 16px;
    }

    .info-text {
      color: rgba(255, 255, 255, 0.8);
      font-size: 14px;
      line-height: 1.6;
      margin: 0;
    }

    .info-text a {
      color: #00d9ff;
      text-decoration: none;
      border-bottom: 1px solid rgba(0, 217, 255, 0.3);
      transition: all 0.2s ease;
    }

    .info-text a:hover {
      color: #00b8e6;
      border-bottom-color: rgba(0, 217, 255, 0.6);
    }

    .status-section {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 16px;
    }

    .status-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .status-item:last-child {
      border-bottom: none;
    }

    .status-label {
      font-weight: 500;
      color: rgba(255, 255, 255, 0.6);
      font-size: 14px;
    }

    .status-value {
      font-weight: 600;
      color: rgba(255, 255, 255, 0.9);
      font-size: 14px;
    }

    .status-value.online {
      color: #22c55e;
    }

    .status-value.offline {
      color: #ef4444;
    }

    .terminal-container {
      flex: 1;
      min-height: 200px;
      background: rgba(0, 0, 0, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 16px;
      font-family: 'Courier New', monospace;
      font-size: 13px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .terminal-line {
      color: rgba(255, 255, 255, 0.8);
      white-space: pre-wrap;
      word-break: break-all;
    }

    .terminal-line.error {
      color: #ef4444;
    }

    .terminal-line.success {
      color: #22c55e;
    }

    .terminal-line.info {
      color: #60a5fa;
    }

    .actions-section {
      display: flex;
      gap: 12px;
      padding-top: 16px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    .start-btn {
      padding: 10px 20px;
      border-radius: 4px;
      border: 1px solid rgba(0, 217, 255, 0.3);
      background: rgba(0, 217, 255, 0.1);
      color: #00d9ff;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .start-btn:hover:not(:disabled) {
      background: rgba(0, 217, 255, 0.2);
      border-color: rgba(0, 217, 255, 0.5);
      transform: scale(1.01);
    }

    .start-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .loading {
      text-align: center;
      padding: 40px 20px;
      color: rgba(255, 255, 255, 0.6);
    }

    .error-message {
      padding: 16px;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      border-radius: 8px;
      color: #ef4444;
      margin-bottom: 16px;
    }

    .empty-state {
      text-align: center;
      padding: 40px 20px;
      color: rgba(255, 255, 255, 0.6);
    }
  `;

  private boundOpen = () => {
    if (!this.isOpen) {
      this.open();
    }
  };

  isModalOpen(): boolean {
    return this.isOpen;
  }

  isModalMinimized(): boolean {
    return this.isMinimized;
  }

  isAcceleratorHealthy(): boolean {
    const statusInfo = this.getStatusInfo();
    return statusInfo.isHealthy;
  }

  connectedCallback() {
    super.connectedCallback();
    document.addEventListener('keydown', this.handleKeyDown);
    document.addEventListener('open-grpc-status-modal', this.boundOpen);
    document.addEventListener('accelerator-unhealthy', this.boundOpen);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    document.removeEventListener('keydown', this.handleKeyDown);
    document.removeEventListener('open-grpc-status-modal', this.boundOpen);
    document.removeEventListener('accelerator-unhealthy', this.boundOpen);
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  }

  private handleKeyDown = (e: KeyboardEvent) => {
    if (this.isOpen && e.key === 'Escape') {
      this.minimize();
    }
  };

  private handleBackdropClick = (e: MouseEvent) => {
    if (e.target === e.currentTarget) {
      this.minimize();
    }
  };

  async open() {
    if (this.isOpen) {
      return;
    }
    
    this.isOpen = true;
    this.setAttribute('open', '');
    requestAnimationFrame(() => {
      const backdrop = this.shadowRoot?.querySelector('.backdrop');
      const modal = this.shadowRoot?.querySelector('.modal');
      if (backdrop) backdrop.classList.add('show');
      if (modal) modal.classList.add('show');
    });
    await Promise.all([this.loadStatus(), this.loadAcceleratorHealth()]);
    this.startAutoRefresh();
  }

  minimize() {
    const backdrop = this.shadowRoot?.querySelector('.backdrop');
    const modal = this.shadowRoot?.querySelector('.modal');
    if (backdrop) backdrop.classList.remove('show');
    if (modal) modal.classList.remove('show');
    
    setTimeout(() => {
      this.isOpen = false;
      this.isMinimized = true;
      this.removeAttribute('open');
    }, 300);
  }

  close() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = null;
    }
    this.minimize();
    this.isMinimized = false;
  }

  restore() {
    if (this.isMinimized) {
      this.isMinimized = false;
      this.open();
    }
  }

  private startAutoRefresh() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
    this.refreshInterval = window.setInterval(() => {
      this.loadAcceleratorHealth();
    }, 5000);
  }

  private async loadStatus() {
    if (this.isLoading) return;

    this.isLoading = true;
    this.error = null;
    this.requestUpdate();

    try {
      await telemetryService.withSpanAsync(
        'grpc.status.check',
        {
          'rpc.service': 'ConfigService',
          'rpc.method': 'getProcessorStatus',
        },
        async (span) => {
          try {
            const response = await this.client.getProcessorStatus({});
            this.status = response;
            this.lastCheck = new Date();
            
            const isAvailable = !!(response.capabilities && response.capabilities.filters);
            span?.setAttribute('grpc.available', isAvailable);
            span?.setAttribute('grpc.api_version', response.apiVersion || 'unknown');
            
            logger.info('gRPC status checked', {
              'grpc.available': isAvailable,
              'grpc.api_version': response.apiVersion,
              'grpc.library_version': response.currentLibrary,
            });
          } catch (error) {
            span?.setAttribute('error', true);
            this.status = null;
            this.lastCheck = new Date();
            
            if (error instanceof ConnectError) {
              span?.setAttribute('error.code', error.code);
              this.error = `Connection error: ${error.message}`;
            } else {
              const errorMsg = error instanceof Error ? error.message : String(error);
              span?.setAttribute('error.message', errorMsg);
              this.error = `Failed to check status: ${errorMsg}`;
            }
            
            logger.error('Failed to check gRPC status', {
              'error.message': error instanceof Error ? error.message : String(error),
            });
          }
        }
      );
    } finally {
      this.isLoading = false;
      this.requestUpdate();
    }
  }

  private async loadAcceleratorHealth() {
    try {
      const response = await remoteManagementService.checkAcceleratorHealth();
      this.acceleratorHealth = response;
      
      logger.info('Accelerator health checked', {
        'accelerator.health.status': response.status,
        'accelerator.server_version': response.serverVersion,
        'accelerator.library_version': response.libraryVersion,
      });
    } catch (error) {
      logger.error('Failed to check accelerator health', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      this.acceleratorHealth = null;
    }
    this.requestUpdate();
  }

  private addTerminalLine(text: string, type: 'info' | 'success' | 'error' | 'normal' = 'normal') {
    const timestamp = new Date().toLocaleTimeString();
    this.terminalOutput.push(`[${timestamp}] ${text}`);
    if (this.terminalOutput.length > 100) {
      this.terminalOutput.shift();
    }
    this.requestUpdate();
    
    const terminalContainer = this.shadowRoot?.querySelector('.terminal-container');
    if (terminalContainer) {
      terminalContainer.scrollTop = terminalContainer.scrollHeight;
    }
  }

  private async handleStartJetson() {
    if (this.isStartingJetson) {
      return;
    }

    this.isStartingJetson = true;
    this.terminalOutput = [];
    this.requestUpdate();

    this.addTerminalLine('Initiating Jetson Nano startup sequence...', 'info');
    this.addTerminalLine('Sending power-on command via MQTT to Tasmota device...', 'info');

    try {
      await remoteManagementService.startJetsonNano(
        (event: StartJetsonNanoEvent) => {
          const status = event.status || 'UNKNOWN';
          const step = event.step || '';
          const message = event.message || '';

          if (step) {
            this.addTerminalLine(`Step: ${step}`, 'info');
          }

          if (message) {
            if (status.includes('ERROR')) {
              this.addTerminalLine(`Error: ${message}`, 'error');
            } else if (status.includes('SUCCESS')) {
              this.addTerminalLine(`Success: ${message}`, 'success');
            } else {
              this.addTerminalLine(message, status.includes('PROGRESS') ? 'info' : 'normal');
            }
          }

          if (status === 'START_JETSON_NANO_STATUS_SUCCESS') {
            this.addTerminalLine('Jetson Nano startup completed successfully!', 'success');
            this.isStartingJetson = false;
            setTimeout(() => {
              this.loadAcceleratorHealth();
            }, 5000);
          } else if (status === 'START_JETSON_NANO_STATUS_ERROR') {
            this.addTerminalLine('Jetson Nano startup failed. Please check the logs.', 'error');
            this.isStartingJetson = false;
          }
        },
        (error: Error) => {
          this.addTerminalLine(`Failed to start Jetson Nano: ${error.message}`, 'error');
          this.isStartingJetson = false;
        }
      );
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      this.addTerminalLine(`Error: ${errorMsg}`, 'error');
      this.isStartingJetson = false;
    }
  }

  private getStatusInfo() {
    if (!this.acceleratorHealth) {
      return {
        isHealthy: false,
        serverVersion: 'Unknown',
        libraryVersion: 'Unknown',
      };
    }

    return {
      isHealthy: this.acceleratorHealth.status === AcceleratorHealthStatus.HEALTHY,
      serverVersion: this.acceleratorHealth.serverVersion || 'Unknown',
      libraryVersion: this.acceleratorHealth.libraryVersion || 'Unknown',
      message: this.acceleratorHealth.message || '',
    };
  }

  render() {
    const statusInfo = this.getStatusInfo();

    return html`
      <div class="backdrop ${this.isOpen ? 'show' : ''}" @click=${this.handleBackdropClick}></div>
      <div class="modal ${this.isOpen ? 'show' : ''}">
        <div class="modal-header">
          <h2 class="modal-title">
            <span class="status-indicator ${statusInfo.isHealthy ? 'online' : 'offline'}"></span>
            CUDA Accelerator Micro Service Status
          </h2>
          <button class="close-btn" @click=${this.minimize} title="Minimize">×</button>
        </div>
        <div class="modal-content">
          <div class="info-section">
            <p class="info-text">
              This project is hosted on a cloud VM without GPU. The NVIDIA GPU is located at my home. 
              To save energy, the GPU is powered off when not in use. I've automated the startup process 
              using a <a href="https://tasmota.github.io/docs/" target="_blank" rel="noopener noreferrer">Tasmota</a>-enabled 
              smart plug (<a href="https://tasmota.github.io/docs/devices/Sonoff-S31/" target="_blank" rel="noopener noreferrer">Sonoff S31</a>) 
              controlled via <a href="https://mqtt.org/" target="_blank" rel="noopener noreferrer">MQTT</a>, integrated with my homelab automation. 
              This interface allows you to start the Jetson Nano remotely. Please note that this is real hardware, 
              and the Jetson Nano's Linux system will take some time to boot up after receiving the power-on command.
            </p>
          </div>

          <div class="terminal-container">
            ${this.terminalOutput.length === 0
              ? html`<div class="empty-state">Terminal output will appear here when starting the Jetson Nano...</div>`
              : this.terminalOutput.map((line) => {
                  let type = 'normal';
                  if (line.includes('Error:') || line.includes('error')) type = 'error';
                  else if (line.includes('Success:') || line.includes('completed')) type = 'success';
                  else if (line.includes('Step:') || line.includes('Initiating') || line.includes('Sending')) type = 'info';
                  
                  return html`<div class="terminal-line ${type}">${line}</div>`;
                })}
          </div>

          <div class="actions-section">
            <button
              class="start-btn"
              @click=${this.handleStartJetson}
              ?disabled=${this.isStartingJetson}
            >
              <span>${this.isStartingJetson ? 'Starting...' : 'Start Jetson Nano'}</span>
            </button>
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'grpc-status-modal': GrpcStatusModal;
  }
}
