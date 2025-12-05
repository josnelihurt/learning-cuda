import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { remoteManagementService, type StartJetsonNanoEvent } from '../../infrastructure/external/remote-management-service';
import { logger } from '../../infrastructure/observability/otel-logger';

interface TerminalLine {
  timestamp: string;
  step: string;
  message: string;
  status: string;
}

@customElement('grpc-unavailable')
export class GrpcUnavailable extends LitElement {
  @state() private terminalLines: TerminalLine[] = [];
  @state() private isStarting = false;

  static styles = css`
    :host {
      display: block;
      width: 100%;
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px;
    }

    .container {
      display: flex;
      flex-direction: column;
      gap: 32px;
    }

    .info-section {
      background: var(--background-secondary, #f5f5f5);
      border-radius: 12px;
      padding: 32px;
      border: 1px solid var(--border-color, #e0e0e0);
    }

    .info-title {
      font-size: 24px;
      font-weight: 600;
      color: var(--text-primary, #333);
      margin: 0 0 16px 0;
    }

    .info-text {
      font-size: 16px;
      line-height: 1.6;
      color: var(--text-secondary, #666);
      margin: 0 0 16px 0;
    }

    .info-text:last-child {
      margin-bottom: 0;
    }

    .infrastructure-list {
      list-style: none;
      padding: 0;
      margin: 16px 0;
    }

    .infrastructure-list li {
      padding: 8px 0;
      padding-left: 24px;
      position: relative;
      color: var(--text-secondary, #666);
    }

    .infrastructure-list li::before {
      content: '•';
      position: absolute;
      left: 8px;
      color: var(--primary-color, #007bff);
      font-weight: bold;
    }

    .action-section {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    .start-button {
      padding: 16px 32px;
      font-size: 16px;
      font-weight: 600;
      background: var(--primary-color, #007bff);
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.2s;
      align-self: flex-start;
    }

    .start-button:hover:not(:disabled) {
      background: var(--primary-color-hover, #0056b3);
    }

    .start-button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .terminal-container {
      background: #1e1e1e;
      border-radius: 8px;
      padding: 20px;
      font-family: 'Courier New', monospace;
      min-height: 300px;
      max-height: 500px;
      overflow-y: auto;
    }

    .terminal-line {
      display: flex;
      gap: 12px;
      padding: 4px 0;
      font-size: 14px;
      line-height: 1.5;
    }

    .terminal-timestamp {
      color: #888;
      flex-shrink: 0;
      width: 80px;
    }

    .terminal-step {
      color: #4ec9b0;
      flex-shrink: 0;
      width: 120px;
      font-weight: 600;
    }

    .terminal-message {
      color: #d4d4d4;
      flex: 1;
    }

    .terminal-status {
      flex-shrink: 0;
      width: 80px;
      text-align: right;
    }

    .status-success {
      color: #4ec9b0;
    }

    .status-error {
      color: #f48771;
    }

    .status-progress {
      color: #dcdcaa;
    }

    .terminal-empty {
      color: #888;
      font-style: italic;
      text-align: center;
      padding: 40px 0;
    }
  `;

  private handleStartJetsonNano = async () => {
    if (this.isStarting) {
      return;
    }

    this.isStarting = true;
    this.terminalLines = [];
    this.addTerminalLine('INIT', 'Starting Jetson Nano initialization process...', 'PROGRESS');

    try {
      await remoteManagementService.startJetsonNano(
        (event: StartJetsonNanoEvent) => {
          this.addTerminalLine(event.step, event.message, event.status);
        },
        (error: Error) => {
          this.addTerminalLine('ERROR', `Failed to start Jetson Nano: ${error.message}`, 'ERROR');
          this.isStarting = false;
        }
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.addTerminalLine('ERROR', `Failed to start Jetson Nano: ${errorMessage}`, 'ERROR');
      logger.error('Failed to start Jetson Nano', {
        'error.message': errorMessage,
      });
    } finally {
      this.isStarting = false;
    }
  };

  private addTerminalLine(step: string, message: string, status: string) {
    const timestamp = new Date().toLocaleTimeString();
    this.terminalLines = [...this.terminalLines, { timestamp, step, message, status }];
    this.requestUpdate();
    
    // Auto-scroll to bottom
    requestAnimationFrame(() => {
      const terminal = this.shadowRoot?.querySelector('.terminal-container');
      if (terminal) {
        terminal.scrollTop = terminal.scrollHeight;
      }
    });
  }

  render() {
    return html`
      <div class="container">
        <div class="info-section">
          <h2 class="info-title">gRPC Server Unavailable</h2>
          <p class="info-text">
            The CUDA accelerator server (gRPC) is currently not available. This server runs on a Jetson Nano device
            located at my home, which is powered off to save energy.
          </p>
          <p class="info-text">
            <strong>Infrastructure Overview:</strong>
          </p>
          <ul class="infrastructure-list">
            <li>
              <strong>Go Web Server:</strong> Running in the cloud on a VM (cost-effective, no GPU)
            </li>
            <li>
              <strong>Jetson Nano:</strong> Located at home with CUDA GPU acceleration (powered off to save energy)
            </li>
            <li>
              <strong>Remote Control:</strong> Using a Sonoff S31 smart switch with Tasmota firmware and MQTT for remote power control
            </li>
          </ul>
          <p class="info-text">
            You can start the Jetson Nano remotely using the button below. The system will send MQTT commands
            to power on the device, wait for it to boot, and establish the gRPC connection.
          </p>
        </div>

        <div class="action-section">
          <button
            class="start-button"
            @click=${this.handleStartJetsonNano}
            ?disabled=${this.isStarting}
          >
            ${this.isStarting ? 'Starting Jetson Nano...' : 'Start Jetson Nano'}
          </button>

          <div class="terminal-container">
            ${this.terminalLines.length === 0
              ? html`<div class="terminal-empty">No activity yet. Click "Start Jetson Nano" to begin.</div>`
              : this.terminalLines.map(
                  (line) => html`
                    <div class="terminal-line">
                      <span class="terminal-timestamp">${line.timestamp}</span>
                      <span class="terminal-step">[${line.step}]</span>
                      <span class="terminal-message">${line.message}</span>
                      <span class="terminal-status status-${line.status.toLowerCase()}">
                        ${line.status}
                      </span>
                    </div>
                  `
                )}
          </div>
        </div>
      </div>
    `;
  }
}


