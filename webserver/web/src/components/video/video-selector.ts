import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { StaticVideo } from '../../gen/common_pb';
import { videoService } from '../../infrastructure/data/video-service';
import { telemetryService } from '../../infrastructure/observability/telemetry-service';
import { logger } from '../../infrastructure/observability/otel-logger';

@customElement('video-selector')
export class VideoSelector extends LitElement {
  @state() private videos: StaticVideo[] = [];
  @state() private selectedVideoId: string | null = null;
  @state() private loading = false;
  @state() private error: string | null = null;

  static styles = css`
    :host {
      display: block;
      padding: 1rem;
    }

    .video-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }

    .video-card {
      border: 2px solid var(--border-color, #ddd);
      border-radius: 8px;
      padding: 0.5rem;
      cursor: pointer;
      transition: all 0.2s;
      background: var(--card-bg, #fff);
    }

    .video-card:hover {
      border-color: var(--primary-color, #007bff);
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .video-card.selected {
      border-color: var(--primary-color, #007bff);
      background: var(--primary-light, #e7f3ff);
    }

    .video-card.default {
      border-color: var(--success-color, #28a745);
    }

    .preview-image {
      width: 100%;
      height: 120px;
      object-fit: cover;
      border-radius: 4px;
      background: var(--preview-bg, #f0f0f0);
    }

    .video-name {
      margin-top: 0.5rem;
      font-weight: 500;
      text-align: center;
      font-size: 0.9rem;
    }

    .default-badge {
      display: inline-block;
      background: var(--success-color, #28a745);
      color: white;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.7rem;
      margin-left: 0.5rem;
    }

    .loading {
      text-align: center;
      padding: 2rem;
      color: var(--text-secondary, #666);
    }

    .error {
      padding: 1rem;
      background: var(--error-bg, #fee);
      color: var(--error-color, #c00);
      border-radius: 4px;
      margin-top: 1rem;
    }

    .no-videos {
      text-align: center;
      padding: 2rem;
      color: var(--text-secondary, #666);
    }
  `;

  async connectedCallback() {
    super.connectedCallback();
    await this.loadVideos();
  }

  private async loadVideos() {
    await telemetryService.withSpanAsync('VideoSelector.loadVideos', {}, async (span) => {
      this.loading = true;
      this.error = null;

      try {
        span?.addEvent('Loading videos');
        this.videos = await videoService.listAvailableVideos();
        span?.setAttribute('videos.count', this.videos.length);

        const defaultVideo = this.videos.find((v) => v.isDefault);
        if (defaultVideo) {
          this.selectedVideoId = defaultVideo.id;
        }
      } catch (err) {
        this.error = 'Failed to load videos';
        span?.setAttribute('error', true);
        logger.error('Error loading videos', {
          'error.message': err instanceof Error ? err.message : String(err),
        });
      } finally {
        this.loading = false;
      }
    });
  }

  private selectVideo(video: StaticVideo) {
    this.selectedVideoId = video.id;

    this.dispatchEvent(
      new CustomEvent('video-selected', {
        detail: { video },
        bubbles: true,
        composed: true,
      })
    );
  }

  render() {
    if (this.loading) {
      return html`<div class="loading">Loading videos...</div>`;
    }

    if (this.error) {
      return html`<div class="error">${this.error}</div>`;
    }

    if (this.videos.length === 0) {
      return html`<div class="no-videos">No videos available</div>`;
    }

    return html`
      <div class="video-grid">
        ${this.videos.map(
          (video) => html`
            <div
              class="video-card ${video.id === this.selectedVideoId
                ? 'selected'
                : ''} ${video.isDefault ? 'default' : ''}"
              @click=${() => this.selectVideo(video)}
              data-testid="video-card-${video.id}"
            >
              <img
                class="preview-image"
                src="${video.previewImagePath || '/static/img/video-placeholder.png'}"
                alt="${video.displayName}"
                loading="lazy"
                @error="${(e: Event) => {
                  (e.target as HTMLImageElement).src = '/static/img/video-placeholder.png';
                }}"
              />
              <div class="video-name">
                ${video.displayName}
                ${video.isDefault ? html`<span class="default-badge">Default</span>` : ''}
              </div>
            </div>
          `
        )}
      </div>
    `;
  }
}
