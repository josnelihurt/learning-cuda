import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { fileService } from '../services/file-service';
import { telemetryService } from '../services/telemetry-service';

const MAX_FILE_SIZE = 10 * 1024 * 1024;

@customElement('image-upload')
export class ImageUpload extends LitElement {
  @state() private isDragging = false;
  @state() private isUploading = false;
  @state() private uploadProgress = 0;
  @state() private error = '';

  static styles = css`
    :host {
      display: block;
      width: 100%;
    }

    .upload-container {
      border: 2px dashed var(--border-color);
      border-radius: 8px;
      padding: 24px;
      text-align: center;
      background: var(--background-secondary);
      transition: all 0.2s;
      cursor: pointer;
    }

    .upload-container:hover {
      border-color: var(--accent-color);
      background: rgba(var(--accent-color-rgb), 0.05);
    }

    .upload-container.dragging {
      border-color: var(--accent-color);
      background: rgba(var(--accent-color-rgb), 0.1);
      transform: scale(1.02);
    }

    .upload-container.uploading {
      cursor: not-allowed;
      opacity: 0.7;
    }

    .upload-icon {
      font-size: 48px;
      color: var(--accent-color);
      margin-bottom: 16px;
    }

    .upload-text {
      color: var(--text-primary);
      font-weight: 600;
      margin-bottom: 8px;
    }

    .upload-hint {
      color: var(--text-secondary);
      font-size: 13px;
      margin-bottom: 4px;
    }

    .upload-format {
      color: var(--text-secondary);
      font-size: 12px;
      font-style: italic;
    }

    .progress-bar {
      width: 100%;
      height: 4px;
      background: var(--background-secondary);
      border-radius: 2px;
      margin-top: 12px;
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      background: var(--accent-color);
      transition: width 0.3s;
    }

    .error {
      color: var(--error-color, #d32f2f);
      font-size: 13px;
      margin-top: 8px;
    }

    input[type='file'] {
      display: none;
    }
  `;

  render() {
    return html`
      <div
        class="upload-container ${this.isDragging ? 'dragging' : ''} ${this.isUploading
          ? 'uploading'
          : ''}"
        @click=${this.handleClick}
        @dragover=${this.handleDragOver}
        @dragleave=${this.handleDragLeave}
        @drop=${this.handleDrop}
        data-testid="upload-container"
      >
        <div class="upload-icon">+</div>
        <div class="upload-text">${this.isUploading ? 'Uploading...' : 'Add Image'}</div>
        <div class="upload-hint">Click or drag and drop to upload</div>
        <div class="upload-format">Only PNG files supported (max 10MB)</div>

        ${this.isUploading
          ? html`
              <div class="progress-bar">
                <div class="progress-fill" style="width: ${this.uploadProgress}%"></div>
              </div>
            `
          : ''}
        ${this.error
          ? html` <div class="error" data-testid="upload-error">${this.error}</div> `
          : ''}
      </div>

      <input type="file" accept=".png" @change=${this.handleFileSelect} data-testid="file-input" />
    `;
  }

  private handleClick(e: Event) {
    if (this.isUploading) {
      return;
    }
    e.preventDefault();
    const input = this.shadowRoot?.querySelector('input[type="file"]') as HTMLInputElement;
    input?.click();
  }

  private handleDragOver(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (!this.isUploading) {
      this.isDragging = true;
    }
  }

  private handleDragLeave(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.isDragging = false;
  }

  private async handleDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    this.isDragging = false;

    if (this.isUploading) {
      return;
    }

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      await this.uploadFile(files[0]);
    }
  }

  private async handleFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const files = input.files;
    if (files && files.length > 0) {
      await this.uploadFile(files[0]);
    }
    input.value = '';
  }

  private async uploadFile(file: File) {
    const span = telemetryService.createSpan('ImageUpload.uploadFile');
    span?.setAttribute('filename', file.name);
    span?.setAttribute('file_size', file.size);

    this.error = '';

    if (!file.name.toLowerCase().endsWith('.png')) {
      this.error = 'Only PNG files are supported';
      span?.setAttribute('validation.error', 'invalid_format');
      span?.end();
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      this.error = 'File size exceeds 10MB limit';
      span?.setAttribute('validation.error', 'file_too_large');
      span?.end();
      return;
    }

    this.isUploading = true;
    this.uploadProgress = 0;

    try {
      const progressInterval = setInterval(() => {
        if (this.uploadProgress < 90) {
          this.uploadProgress += 10;
        }
      }, 100);

      const image = await fileService.uploadImage(file);

      clearInterval(progressInterval);
      this.uploadProgress = 100;

      setTimeout(() => {
        this.isUploading = false;
        this.uploadProgress = 0;

        this.dispatchEvent(
          new CustomEvent('image-uploaded', {
            bubbles: true,
            composed: true,
            detail: { image },
          })
        );

        span?.setAttribute('upload.status', 'success');
        span?.setAttribute('image.id', image.id);
        span?.end();

        logger.info('Image upload complete', {
          'image.id': image.id,
        });
      }, 300);
    } catch (error) {
      this.isUploading = false;
      this.uploadProgress = 0;
      this.error = error instanceof Error ? error.message : 'Upload failed';

      span?.recordException(error as Error);
      span?.setAttribute('upload.status', 'failed');
      span?.end();

      logger.error('Image upload failed', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'image-upload': ImageUpload;
  }
}
