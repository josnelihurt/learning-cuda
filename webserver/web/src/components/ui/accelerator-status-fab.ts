import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { logger } from '../../infrastructure/observability/otel-logger';
import { remoteManagementService } from '../../infrastructure/external/remote-management-service';
import { AcceleratorHealthStatus } from '../../gen/remote_management_service_pb';

@customElement('accelerator-status-fab')
export class AcceleratorStatusFab extends LitElement {
  @state() private isHealthy = false;
  @state() private isBlinking = false;
  private checkInterval: number | null = null;

  static styles = css`
    :host {
      position: fixed;
      bottom: 160px;
      right: 32px;
      z-index: 1000;
    }

    :host([hidden]) {
      display: none !important;
    }

    .fab {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px 24px;
      background: #ef4444;
      color: white;
      border: none;
      border-radius: 28px;
      box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
      cursor: pointer;
      font-size: 16px;
      font-weight: 600;
      transition: all 0.2s;
    }

    .fab.blinking {
      animation: heartbeat 2s ease-in-out infinite;
    }

    @keyframes heartbeat {
      0% {
        transform: scale(1);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
      }
      14% {
        transform: scale(1.1);
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.6);
      }
      28% {
        transform: scale(1);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
      }
      42% {
        transform: scale(1.1);
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.6);
      }
      70% {
        transform: scale(1);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
      }
      100% {
        transform: scale(1);
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
      }
    }

    .fab:hover:not(.blinking) {
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(239, 68, 68, 0.5);
    }

    .fab:active {
      transform: translateY(0);
    }

    .fab-icon {
      font-size: 24px;
      line-height: 1;
    }

    @media (max-width: 768px) {
      :host {
        bottom: 160px;
        right: 32px;
      }

      .fab-text {
        display: none;
      }

      .fab {
        padding: 16px;
        border-radius: 50%;
        width: 56px;
        height: 56px;
        justify-content: center;
      }
    }
  `;

  connectedCallback() {
    super.connectedCallback();
    this.setAttribute('hidden', '');
    this.checkHealth();
    this.startChecking();
    document.addEventListener('accelerator-health-changed', this.handleHealthChange);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.stopChecking();
    document.removeEventListener('accelerator-health-changed', this.handleHealthChange);
  }

  private handleHealthChange = (e: Event) => {
    const customEvent = e as CustomEvent;
    this.isHealthy = customEvent.detail?.healthy ?? false;
    this.isBlinking = !this.isHealthy;
    this.requestUpdate();
  };

  private startChecking() {
    this.checkHealth();
    this.checkInterval = window.setInterval(() => {
      this.checkHealth();
    }, 5000);
  }

  private stopChecking() {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  private async checkHealth() {
    const modalElement = document.querySelector('grpc-status-modal') as any;
    const isModalOpen = modalElement && modalElement.isModalOpen && modalElement.isModalOpen();

    try {
      const response = await remoteManagementService.checkAcceleratorHealth();
      const isHealthy = response.status === AcceleratorHealthStatus.HEALTHY;

      this.isHealthy = isHealthy;
      this.isBlinking = !isHealthy;

      const shouldShow = !isHealthy && !isModalOpen;

      if (shouldShow) {
        this.removeAttribute('hidden');
      } else {
        this.setAttribute('hidden', '');
      }
    } catch (error) {
      logger.error('Failed to check accelerator health in FAB', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      this.isHealthy = false;
      this.isBlinking = true;
      
      if (!isModalOpen) {
        this.removeAttribute('hidden');
      } else {
        this.setAttribute('hidden', '');
      }
    }

    this.requestUpdate();
  }

  private handleClick(): void {
    logger.debug('Accelerator status FAB clicked - opening modal');
    
    const modalElement = document.querySelector('grpc-status-modal') as any;
    if (modalElement) {
      if (modalElement.isModalMinimized && modalElement.isModalMinimized()) {
        modalElement.restore();
      } else {
        modalElement.open();
      }
    }
  }

  render() {
    return html`
      <button 
        class="fab ${this.isBlinking ? 'blinking' : ''}" 
        @click=${this.handleClick}
        data-testid="accelerator-status-fab"
        title="Accelerator offline - Click to view status"
      >
        <span class="fab-icon">!</span>
        <span class="fab-text">Accelerator Offline</span>
      </button>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'accelerator-status-fab': AcceleratorStatusFab;
  }
}

