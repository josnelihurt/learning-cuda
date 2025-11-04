import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { InputSource } from '../gen/config_service_pb';
import { WebSocketService } from '../services/websocket-service';
import { telemetryService } from '../services/telemetry-service';
import { processorCapabilitiesService } from '../services/processor-capabilities-service';
import type { StatsPanel } from './stats-panel';
import type { CameraPreview } from './camera-preview';
import type { ToastContainer } from './toast-container';
import './video-source-card';

interface GridSource {
  id: string;
  number: number;
  name: string;
  type: string;
  imagePath: string;
  originalImageSrc: string;
  currentImageSrc: string;
  ws: WebSocketService | null;
  cameraPreview: CameraPreview | null;
  filters: string[];
  grayscaleType: string;
  resolution: string;
  blurParams?: Record<string, any>;
  videoId?: string;
}

@customElement('video-grid')
export class VideoGrid extends LitElement {
  @state() private sources: GridSource[] = [];
  @state() private selectedSourceId: string | null = null;

  private statsManager: StatsPanel | null = null;
  private toastManager: ToastContainer | null = null;
  private nextNumber = 1;
  private readonly MAX_SOURCES = 9;

  static styles = css`
    :host {
      display: block;
      width: 100%;
      height: 100%;
      padding: var(--spacing-md);
      overflow: hidden;
      box-sizing: border-box;
    }

    .grid-container {
      display: grid;
      gap: 0;
      width: 100%;
      height: 100%;
    }

    /* Grid layouts that scale to fit viewport - NO SCROLLBARS */
    .grid-1 {
      grid-template-columns: 1fr;
      grid-template-rows: 1fr;
    }
    .grid-2 {
      grid-template-columns: 1fr;
      grid-template-rows: repeat(2, 1fr);
    }
    .grid-3,
    .grid-4 {
      grid-template-columns: repeat(2, 1fr);
      grid-template-rows: repeat(2, 1fr);
    }
    .grid-5,
    .grid-6 {
      grid-template-columns: repeat(3, 1fr);
      grid-template-rows: repeat(2, 1fr);
    }
    .grid-7,
    .grid-8,
    .grid-9 {
      grid-template-columns: repeat(3, 1fr);
      grid-template-rows: repeat(3, 1fr);
    }
  `;

  render() {
    const gridClass = `grid-${this.sources.length}`;

    return html`
      <div class="grid-container ${gridClass}" data-testid="video-grid">
        ${this.sources.map((source) => this.renderSourceCard(source))}
      </div>
    `;
  }

  private renderSourceCard(source: GridSource) {
    return html`
      <video-source-card
        .sourceId=${source.id}
        .sourceNumber=${source.number}
        .sourceName=${source.name}
        .sourceType=${source.type}
        .imageSrc=${source.currentImageSrc || source.imagePath}
        .isSelected=${this.selectedSourceId === source.id}
        @source-selected=${() => this.selectSource(source.id)}
        @source-closed=${() => this.removeSource(source.id)}
        data-testid="video-source-card"
        data-source-id="${source.id}"
      >
        ${source.cameraPreview || ''}
      </video-source-card>
    `;
  }

  setManagers(statsManager: StatsPanel, toastManager: ToastContainer): void {
    this.statsManager = statsManager;
    this.toastManager = toastManager;
  }

  addSource(inputSource: InputSource): boolean {
    if (this.sources.length >= this.MAX_SOURCES) {
      this.toastManager?.warning(
        'Maximum Sources',
        `Cannot add more than ${this.MAX_SOURCES} sources`
      );
      return false;
    }

    const number = this.nextNumber++;
    const uniqueId = `${inputSource.id}-${number}`;

    let cameraPreview: CameraPreview | null = null;
    let ws: WebSocketService | null = null;

    if (inputSource.type === 'camera') {
      cameraPreview = document.createElement('camera-preview') as CameraPreview;
      if (this.statsManager && this.toastManager) {
        cameraPreview.setManagers(this.statsManager, this.toastManager);
      }

      ws = new WebSocketService(this.statsManager!, cameraPreview, this.toastManager!);
      ws.connect();

      ws.onFrameResult((data) => {
        const sourceIndex = this.sources.findIndex((s) => s.number === number);
        if (sourceIndex !== -1 && data.response) {
          let binary = '';
          const len = data.response.imageData.byteLength;
          for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(data.response.imageData[i]);
          }
          const imageData = btoa(binary);
          this.sources[sourceIndex].currentImageSrc = `data:image/png;base64,${imageData}`;
          this.sources = [...this.sources];
          this.requestUpdate();
        }
      });

      setTimeout(async () => {
        if (cameraPreview) {
          const success = await cameraPreview.start();
          if (success && ws) {
            cameraPreview.startCapture((base64Data: string, width: number, height: number) => {
              // Add data URI prefix since ImageData expects it
              const dataUri = `data:image/jpeg;base64,${base64Data}`;
              // Use current filters from the source
              const sourceIndex = this.sources.findIndex((s) => s.id === uniqueId);
              const source = sourceIndex !== -1 ? this.sources[sourceIndex] : null;
              const filters = source?.filters || ['none'];
              const grayscaleType = source?.grayscaleType || 'bt601';
              ws!.sendFrame(dataUri, width, height, filters, 'gpu', grayscaleType);
            });
          }
        }
      }, 500);
    } else if (inputSource.type === 'video') {
      ws = new WebSocketService(
        this.statsManager!,
        document.createElement('camera-preview') as CameraPreview,
        this.toastManager!
      );
      ws.connect();

      ws.onFrameResult((data) => {
        const sourceIndex = this.sources.findIndex((s) => s.number === number);
        if (sourceIndex !== -1) {
          const frameData = data.videoFrame || data.response;
          if (!frameData) return;

          let binary = '';
          const len = frameData.imageData?.byteLength || frameData.frameData?.byteLength || 0;
          const imageBytes = frameData.imageData || frameData.frameData;
          if (!imageBytes) return;

          for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(imageBytes[i]);
          }
          const imageData = btoa(binary);
          const newSrc = `data:image/png;base64,${imageData}`;
          this.sources[sourceIndex].currentImageSrc = newSrc;
          this.sources = [...this.sources];
          this.requestUpdate();
        }
      });

      const tryStartVideo = () => {
        if (ws!.isConnected()) {
          logger.debug('WebSocket is connected, starting video', {
            'video.id': inputSource.id,
          });
          ws!.sendStartVideo(inputSource.id, ['none'], 'gpu', 'bt601');
        } else {
          logger.debug('WebSocket not ready, retrying in 100ms...');
          setTimeout(tryStartVideo, 100);
        }
      };
      setTimeout(tryStartVideo, 100);
    } else {
      ws = new WebSocketService(
        this.statsManager!,
        document.createElement('camera-preview') as CameraPreview,
        this.toastManager!
      );
      ws.connect();

      ws.onFrameResult((data) => {
        logger.debug('Frame result for source', {
          'source.number': number,
          'frame.success': data.success,
        });
        const sourceIndex = this.sources.findIndex((s) => s.number === number);
        if (sourceIndex !== -1 && data.response) {
          let binary = '';
          const len = data.response.imageData.byteLength;
          for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(data.response.imageData[i]);
          }
          const imageData = btoa(binary);
          const newSrc = `data:image/png;base64,${imageData}`;
          logger.debug('Updating image for source', {
            'source.number': number,
          });
          this.sources[sourceIndex].currentImageSrc = newSrc;
          this.sources = [...this.sources];
          this.requestUpdate();
        }
      });
    }

    const sourceImagePath =
      inputSource.type === 'video' ? inputSource.previewImagePath || '' : inputSource.imagePath;

    const newSource: GridSource = {
      id: uniqueId,
      number,
      name: inputSource.displayName,
      type: inputSource.type,
      imagePath: sourceImagePath,
      originalImageSrc: sourceImagePath,
      currentImageSrc: sourceImagePath,
      ws,
      cameraPreview,
      filters: [],
      grayscaleType: 'bt601',
      resolution: 'original',
      videoId: inputSource.type === 'video' ? inputSource.id : undefined,
    };

    this.sources = [...this.sources, newSource];

    if (this.sources.length === 1) {
      this.selectSource(uniqueId);
    }

    logger.debug(`Source added to grid. Total: ${this.sources.length}`, {
      'source.id': uniqueId,
      'grid.total': this.sources.length,
    });

    this.requestUpdate();
    return true;
  }

  removeSource(sourceId: string): void {
    const source = this.sources.find((s) => s.id === sourceId);
    if (!source) return;

    if (source.ws) {
      source.ws.disconnect();
    }

    if (source.cameraPreview) {
      source.cameraPreview.stop();
    }

    this.sources = this.sources.filter((s) => s.id !== sourceId);

    if (this.selectedSourceId === sourceId && this.sources.length > 0) {
      this.selectSource(this.sources[0].id);
    }

    logger.debug(`Source removed. Remaining: ${this.sources.length}`, {
      'source.id': sourceId,
      'grid.remaining': this.sources.length,
    });
  }

  changeSourceImage(sourceNumber: number, newImagePath: string): void {
    const sourceIndex = this.sources.findIndex((s) => s.number === sourceNumber);
    if (sourceIndex === -1) {
      logger.error('Source not found', {
        'source.number': sourceNumber,
      });
      return;
    }

    if (this.sources[sourceIndex].type === 'camera') {
      logger.warn('Cannot change image of camera source');
      return;
    }

    this.sources[sourceIndex].imagePath = newImagePath;
    this.sources[sourceIndex].originalImageSrc = newImagePath;
    this.sources[sourceIndex].currentImageSrc = newImagePath;
    this.sources = [...this.sources];
    this.requestUpdate();
    logger.debug('Source image changed', {
      'source.number': sourceNumber,
      'image.path': newImagePath,
    });
  }

  selectSource(sourceId: string): void {
    const source = this.sources.find((s) => s.id === sourceId);
    if (!source) return;

    this.selectedSourceId = sourceId;

    logger.debug('Card selected', {
      'source.number': source.number,
      'source.id': sourceId,
    });

    this.dispatchEvent(
      new CustomEvent('source-selection-changed', {
        bubbles: true,
        composed: true,
        detail: {
          sourceId,
          sourceNumber: source.number,
          sourceType: source.type,
          filters: source.filters,
          grayscaleType: source.grayscaleType,
          resolution: source.resolution,
          blurParams: source.blurParams,
        },
      })
    );
  }

  getSelectedSource(): GridSource | null {
    return this.sources.find((s) => s.id === this.selectedSourceId) || null;
  }

  getSelectedSourceIds(): Set<string> {
    return new Set(this.sources.map((s) => s.id));
  }

  getSources(): GridSource[] {
    return this.sources;
  }

  async applyFilterToSelected(
    filters: string[],
    accelerator: string,
    grayscaleType: string,
    resolution: string = 'original',
    blurParams?: Record<string, any>
  ): Promise<void> {
    const selectedSource = this.getSelectedSource();
    if (!selectedSource) {
      logger.debug('applyFilter: no source selected');
      return;
    }

    selectedSource.filters = filters;
    selectedSource.grayscaleType = grayscaleType;
    selectedSource.resolution = resolution;
    selectedSource.blurParams = blurParams;

    logger.debug(`Applying filter to source ${selectedSource.number} ${grayscaleType}`, {
      'source.number': selectedSource.number,
      'source.type': selectedSource.type,
      filters: filters.join(','),
      grayscale_type: grayscaleType,
      resolution: resolution,
    });

    if (!selectedSource.ws || !selectedSource.ws.isConnected()) {
      logger.error('WebSocket not connected for source', {
        'source.number': selectedSource.number,
      });
      return;
    }

    if (selectedSource.type === 'video') {
      try {
        const videoId = selectedSource.videoId || selectedSource.name;
        logger.debug('Restarting video with new filters', {
          'video.id': videoId,
          filters: filters.join(','),
          accelerator: accelerator,
          grayscale_type: grayscaleType,
        });
        selectedSource.ws.sendStopVideo(videoId);

        setTimeout(() => {
          if (selectedSource.ws && selectedSource.ws.isConnected()) {
            logger.debug('Starting video with filters', {
              filters: filters.join(','),
            });
            selectedSource.ws.sendStartVideo(videoId, filters, accelerator, grayscaleType, blurParams);
          }
        }, 200);
      } catch (error) {
        logger.error('Error updating video filters', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
        this.toastManager?.error('Filter Error', 'Failed to update video filters');
      }
      return;
    }

    if (selectedSource.type === 'camera') {
      logger.debug('Camera filter update not yet implemented');
      return;
    }

    try {
      const originalImg = new Image();
      originalImg.crossOrigin = 'anonymous';

      await new Promise<void>((resolve, reject) => {
        originalImg.onload = () => resolve();
        originalImg.onerror = () => reject(new Error('Failed to load original image'));
        originalImg.src = selectedSource.originalImageSrc;
      });

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const originalWidth = originalImg.naturalWidth || originalImg.width || 512;
      const originalHeight = originalImg.naturalHeight || originalImg.height || 512;

      let targetWidth = originalWidth;
      let targetHeight = originalHeight;

      switch (resolution) {
        case 'half':
          targetWidth = Math.floor(originalWidth / 2);
          targetHeight = Math.floor(originalHeight / 2);
          break;
        case 'quarter':
          targetWidth = Math.floor(originalWidth / 4);
          targetHeight = Math.floor(originalHeight / 4);
          break;
        case 'original':
        default:
          break;
      }

      canvas.width = targetWidth;
      canvas.height = targetHeight;

      ctx.drawImage(originalImg, 0, 0, targetWidth, targetHeight);
      const imageData = canvas.toDataURL('image/png');

      logger.debug(`Sending image ${targetWidth} x ${targetHeight}`, {
        'image.original_width': originalWidth,
        'image.original_height': originalHeight,
        'image.target_width': targetWidth,
        'image.target_height': targetHeight,
      });

      const response = await selectedSource.ws.sendSingleFrame(
        imageData,
        targetWidth,
        targetHeight,
        filters,
        accelerator,
        grayscaleType,
        blurParams
      );

      if (response.success && response.response) {
        logger.debug('Filter applied, updating image for source', {
          'source.number': selectedSource.number,
        });
        let binary = '';
        const len = response.response.imageData.byteLength;
        for (let i = 0; i < len; i++) {
          binary += String.fromCharCode(response.response.imageData[i]);
        }
        const processedImageData = btoa(binary);
        const sourceIndex = this.sources.findIndex((s) => s.id === selectedSource.id);
        if (sourceIndex !== -1) {
          this.sources[sourceIndex].currentImageSrc = `data:image/png;base64,${processedImageData}`;
          this.sources = [...this.sources];
          this.requestUpdate();
        }
      }
    } catch (error) {
      logger.error('Error applying filter', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      this.toastManager?.error('Filter Error', 'Failed to apply filter');
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'video-grid': VideoGrid;
  }
}
