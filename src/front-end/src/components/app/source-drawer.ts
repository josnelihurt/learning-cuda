import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { InputSource } from '../../gen/config_service_pb';
import { telemetryService } from '../../infrastructure/observability/telemetry-service';
import { logger } from '../../infrastructure/observability/otel-logger';
import '../image/image-upload';
import '../video/video-upload';
import '../video/video-selector';

@customElement('source-drawer')
export class SourceDrawer extends LitElement {
  @state() private isOpen = false;
  @state() private availableSources: InputSource[] = [];
  @state() private selectedSourceIds: Set<string> = new Set();
  @state() private activeTab: 'images' | 'videos' = 'images';

  static styles = css`
    :host {
      position: fixed;
      top: 0;
      right: 0;
      bottom: 0;
      width: 100%;
      z-index: 9999;
      pointer-events: none;
    }

    .backdrop {
      position: absolute;
      inset: 0;
      background: rgba(0, 0, 0, 0.5);
      opacity: 0;
      transition: opacity 0.3s ease;
      pointer-events: none;
    }

    .backdrop.show {
      opacity: 1;
      pointer-events: auto;
    }

    .drawer {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      width: 400px;
      max-width: 90vw;
      background: white;
      box-shadow: -4px 0 24px rgba(0, 0, 0, 0.15);
      transform: translateX(100%);
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      flex-direction: column;
      pointer-events: auto;
    }

    .drawer.show {
      transform: translateX(0);
    }

    .drawer-header {
      padding: 24px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .drawer-title {
      font-size: 20px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .close-btn {
      background: none;
      border: none;
      font-size: 24px;
      cursor: pointer;
      color: var(--text-secondary);
      padding: 4px;
      line-height: 1;
      transition: color 0.2s;
    }

    .close-btn:hover {
      color: var(--text-primary);
    }

    .drawer-content {
      flex: 1;
      overflow-y: auto;
      padding: 16px 24px;
    }

    .source-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .source-item {
      padding: 16px;
      border: 2px solid var(--border-color);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .source-item:hover {
      border-color: var(--accent-color);
      background: var(--background-secondary);
    }

    .source-item.disabled {
      opacity: 0.5;
      cursor: not-allowed;
      background: var(--background-secondary);
    }

    .source-item.disabled:hover {
      border-color: var(--border-color);
    }

    .source-icon {
      font-size: 24px;
      flex-shrink: 0;
    }

    .source-info {
      flex: 1;
    }

    .source-name {
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 4px;
    }

    .source-type {
      font-size: 13px;
      color: var(--text-secondary);
      text-transform: capitalize;
    }

    .source-badge {
      background: var(--accent-color);
      color: white;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
    }

    .upload-section {
      padding: 0 0 16px 0;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 16px;
    }

    .section-title {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .tabs {
      display: flex;
      border-bottom: 2px solid var(--border-color);
      margin-bottom: 16px;
    }

    .tab {
      flex: 1;
      padding: 12px;
      background: none;
      border: none;
      cursor: pointer;
      font-weight: 500;
      color: var(--text-secondary);
      transition: all 0.2s;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
    }

    .tab:hover {
      color: var(--text-primary);
    }

    .tab.active {
      color: var(--primary-color);
      border-bottom-color: var(--primary-color);
    }
  `;

  render() {
    return html`
      <div class="backdrop ${this.isOpen ? 'show' : ''}" @click=${this.close}></div>
      <div class="drawer ${this.isOpen ? 'show' : ''}" data-testid="source-drawer">
        <div class="drawer-header">
          <h2 class="drawer-title">Select Input Source</h2>
          <button class="close-btn" @click=${this.close} data-testid="drawer-close">×</button>
        </div>
        <div class="drawer-content">
          <div class="tabs">
            <button
              class="tab ${this.activeTab === 'images' ? 'active' : ''}"
              @click=${() => (this.activeTab = 'images')}
              data-testid="tab-images"
            >
              Images
            </button>
            <button
              class="tab ${this.activeTab === 'videos' ? 'active' : ''}"
              @click=${() => (this.activeTab = 'videos')}
              data-testid="tab-videos"
            >
              Videos
            </button>
          </div>

          ${this.activeTab === 'images'
            ? html`
                <div class="upload-section">
                  <div class="section-title">Upload Image</div>
                  <image-upload @image-uploaded=${this.handleImageUploaded}></image-upload>
                </div>
              `
            : html`
                <div class="upload-section">
                  <div class="section-title">Upload Video</div>
                  <video-upload @video-uploaded=${this.handleVideoUploaded}></video-upload>
                </div>
              `}
          ${this.activeTab === 'images'
            ? html`
                <div class="section-title">Select Source</div>
                <div class="source-list">
                  ${this.availableSources
                    .filter((s) => s.type === 'static' || s.type === 'camera')
                    .map((source) => this.renderSourceItem(source))}
                </div>
              `
            : html`
                <div class="section-title">Select Video</div>
                <video-selector @video-selected=${this.handleVideoSelected}></video-selector>
              `}
        </div>
      </div>
    `;
  }

  private renderSourceItem(source: InputSource) {
    const icon = source.type === 'camera' ? '●' : source.type === 'video' ? '▶' : '▣';

    return html`
      <div
        class="source-item"
        @click=${() => this.selectSource(source)}
        data-testid="source-item-${source.id}"
      >
        <div class="source-icon">${icon}</div>
        <div class="source-info">
          <div class="source-name">${source.displayName}</div>
          <div class="source-type">${source.type}</div>
        </div>
        ${source.isDefault ? html`<span class="source-badge">Default</span>` : ''}
      </div>
    `;
  }

  open(sources: InputSource[], selectedIds: Set<string>): void {
    logger.debug('Drawer opened', {
      'sources.available': sources.length,
      'sources.selected': selectedIds.size,
    });

    this.availableSources = sources;
    this.selectedSourceIds = selectedIds;
    this.isOpen = true;
  }

  close(): void {
    this.isOpen = false;
  }

  private selectSource(source: InputSource): void {
    logger.debug(`Source selected: ${source.id}`, {
      'source.id': source.id,
      'source.type': source.type,
    });

    this.dispatchEvent(
      new CustomEvent('source-selected', {
        bubbles: true,
        composed: true,
        detail: { source },
      })
    );

    this.close();
  }

  private handleImageUploaded(event: CustomEvent): void {
    const span = telemetryService.createSpan('SourceDrawer.handleImageUploaded');
    const { image } = event.detail;

    logger.info('Image uploaded in drawer', {
      'image.id': image.id,
    });
    span?.setAttribute('image.id', image.id);

    this.dispatchEvent(
      new CustomEvent('image-uploaded', {
        bubbles: true,
        composed: true,
        detail: { image },
      })
    );

    span?.end();
  }

  private handleVideoUploaded(event: CustomEvent): void {
    const span = telemetryService.createSpan('SourceDrawer.handleVideoUploaded');
    const { video } = event.detail;

    logger.info('Video uploaded in drawer', {
      'video.id': video.id,
    });
    span?.setAttribute('video.id', video.id);

    this.dispatchEvent(
      new CustomEvent('video-uploaded', {
        bubbles: true,
        composed: true,
        detail: { video },
      })
    );

    span?.end();
  }

  private handleVideoSelected(event: CustomEvent): void {
    const span = telemetryService.createSpan('SourceDrawer.handleVideoSelected');
    const { video } = event.detail;

    logger.debug('Video selected in drawer', {
      'video.id': video.id,
    });
    span?.setAttribute('video.id', video.id);

    const videoSource: InputSource = {
      id: video.id,
      displayName: video.displayName,
      type: 'video',
      imagePath: '',
      isDefault: video.isDefault,
      videoPath: video.path,
      previewImagePath: video.previewImagePath,
    };

    this.selectSource(videoSource);

    span?.end();
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'source-drawer': SourceDrawer;
  }
}
