import { LitElement, html, css } from 'lit';
import { customElement, property, query, state } from 'lit/decorators.js';
import { logger } from '../services/otel-logger';

interface StatsManager {
  updateCameraStatus(status: string, type: 'success' | 'error' | 'warning' | 'inactive'): void;
}

interface ToastManager {
  error(title: string, message: string): void;
}

type FrameCallback = (base64data: string, width: number, height: number, timestamp: number) => void;

@customElement('camera-preview')
export class CameraPreview extends LitElement {
  @property({ type: Number }) width = 640;
  @property({ type: Number }) height = 480;
  @property({ type: Number }) fps = 15;
  @property({ type: Number }) quality = 0.7;
  @property({ type: String, attribute: false }) heroSrc = '';

  @query('video') private videoElement!: HTMLVideoElement;
  @query('canvas') private canvasElement!: HTMLCanvasElement;

  @state() private isProcessing = false;

  private statsManager: StatsManager | null = null;
  private toastManager: ToastManager | null = null;
  private stream: MediaStream | null = null;
  private frameInterval: number | null = null;
  private lastFrameTime = 0;
  private onFrameCallback: FrameCallback | null = null;

  static styles = css`
    :host {
      display: contents;
    }

    video {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }

    canvas {
      display: none;
    }
  `;

  render() {
    return html`
      <slot name="hero-image"></slot>
      <video autoplay playsinline muted></video>
      <canvas></canvas>
    `;
  }

  setManagers(statsManager: StatsManager, toastManager: ToastManager): void {
    this.statsManager = statsManager;
    this.toastManager = toastManager;
  }

  setResolution(width: number, height: number): void {
    this.width = width;
    this.height = height;

    if (this.frameInterval) {
      this.stopCapture();
      this.startCapture(this.onFrameCallback!);
    }
  }

  async start(): Promise<boolean> {
    try {
      this.statsManager?.updateCameraStatus('Requesting access...', 'warning');

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera API not available. Use HTTPS.');
      }

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
      });

      this.videoElement.srcObject = this.stream;
      await new Promise<void>((resolve) => (this.videoElement.onloadedmetadata = () => resolve()));
      await this.videoElement.play();
      await new Promise<void>((resolve) => setTimeout(resolve, 500));

      logger.info('Camera ready', {
        'camera.width': this.videoElement.videoWidth,
        'camera.height': this.videoElement.videoHeight,
      });

      this.statsManager?.updateCameraStatus('Active', 'success');

      this.dispatchEvent(
        new CustomEvent('camera-started', {
          bubbles: true,
          composed: true,
        })
      );

      return true;
    } catch (error) {
      logger.error('Camera error', {
        'error.message': error instanceof Error ? error.message : String(error),
      });

      let errorTitle = 'Camera Error';
      let errorMsg = '';

      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          errorTitle = 'Permission Denied';
          errorMsg = 'Please allow camera access in your browser settings';
        } else if (error.name === 'NotFoundError') {
          errorTitle = 'No Camera Found';
          errorMsg = 'No camera device detected on this system';
        } else if (error.name === 'NotReadableError') {
          errorTitle = 'Camera In Use';
          errorMsg = 'Camera is being used by another application. Please close it and try again';
        } else if (error.message) {
          errorMsg = error.message;
        }
      }

      this.toastManager?.error(errorTitle, errorMsg);

      if (this.statsManager) {
        const shortMsg = errorTitle.replace(' Error', '').substring(0, 20);
        this.statsManager.updateCameraStatus(shortMsg, 'error');
      }

      this.dispatchEvent(
        new CustomEvent('camera-error', {
          bubbles: true,
          composed: true,
          detail: { error, errorTitle, errorMsg },
        })
      );

      return false;
    }
  }

  stop(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    this.stopCapture();

    if (this.videoElement?.srcObject) {
      this.videoElement.srcObject = null;
    }

    this.statsManager?.updateCameraStatus('Inactive', 'inactive');
  }

  startCapture(onFrameCallback: FrameCallback): void {
    if (!this.canvasElement) return;

    this.onFrameCallback = onFrameCallback;
    const ctx = this.canvasElement.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    this.canvasElement.width = this.width;
    this.canvasElement.height = this.height;

    logger.info('Starting capture', {
      'capture.width': this.width,
      'capture.height': this.height,
    });

    this.frameInterval = window.setInterval(() => {
      if (!this.videoElement.videoWidth || this.isProcessing) return;

      this.lastFrameTime = performance.now();

      ctx.drawImage(this.videoElement, 0, 0, this.width, this.height);
      const dataUrl = this.canvasElement.toDataURL('image/jpeg', this.quality);
      const base64data = dataUrl.split(',')[1];

      if (this.onFrameCallback) {
        this.onFrameCallback(base64data, this.width, this.height, this.lastFrameTime);
      }

      this.dispatchEvent(
        new CustomEvent('frame-captured', {
          bubbles: true,
          composed: true,
          detail: {
            base64data,
            width: this.width,
            height: this.height,
            timestamp: this.lastFrameTime,
          },
        })
      );
    }, 1000 / this.fps);
  }

  stopCapture(): void {
    if (this.frameInterval !== null) {
      clearInterval(this.frameInterval);
      this.frameInterval = null;
    }
  }

  setProcessing(isProcessing: boolean): void {
    this.isProcessing = isProcessing;
  }

  getLastFrameTime(): number {
    return this.lastFrameTime;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'camera-preview': CameraPreview;
  }
}
