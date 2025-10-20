import { LitElement, html, css } from 'lit';
import { customElement, state } from 'lit/decorators.js';
import { videoService } from '../services/video-service';
import { telemetryService } from '../services/telemetry-service';

@customElement('video-upload')
export class VideoUpload extends LitElement {
    @state() private uploading = false;
    @state() private uploadProgress = 0;
    @state() private error: string | null = null;
    @state() private success: string | null = null;

    static styles = css`
        :host {
            display: block;
            padding: 1rem;
        }

        .upload-container {
            border: 2px dashed var(--border-color, #ddd);
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            transition: border-color 0.2s;
        }

        .upload-container:hover {
            border-color: var(--primary-color, #007bff);
        }

        .upload-container.dragging {
            border-color: var(--primary-color, #007bff);
            background: var(--primary-light, #e7f3ff);
        }

        input[type="file"] {
            display: none;
        }

        .upload-button {
            padding: 0.75rem 1.5rem;
            background: var(--primary-color, #007bff);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1rem;
            transition: background 0.2s;
        }

        .upload-button:hover:not(:disabled) {
            background: var(--primary-dark, #0056b3);
        }

        .upload-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .upload-info {
            margin-top: 1rem;
            color: var(--text-secondary, #666);
            font-size: 0.9rem;
        }

        .progress-bar {
            width: 100%;
            height: 4px;
            background: var(--progress-bg, #e0e0e0);
            border-radius: 2px;
            margin-top: 1rem;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--primary-color, #007bff);
            transition: width 0.3s;
        }

        .message {
            padding: 1rem;
            border-radius: 4px;
            margin-top: 1rem;
        }

        .error {
            background: var(--error-bg, #fee);
            color: var(--error-color, #c00);
        }

        .success {
            background: var(--success-bg, #efe);
            color: var(--success-color, #090);
        }
    `;

    private handleFileSelect(event: Event) {
        const input = event.target as HTMLInputElement;
        if (input.files && input.files.length > 0) {
            this.uploadVideo(input.files[0]);
        }
    }

    private handleDrop(event: DragEvent) {
        event.preventDefault();
        const container = event.currentTarget as HTMLElement;
        container.classList.remove('dragging');

        if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
            this.uploadVideo(event.dataTransfer.files[0]);
        }
    }

    private handleDragOver(event: DragEvent) {
        event.preventDefault();
        const container = event.currentTarget as HTMLElement;
        container.classList.add('dragging');
    }

    private handleDragLeave(event: DragEvent) {
        const container = event.currentTarget as HTMLElement;
        container.classList.remove('dragging');
    }

    private async uploadVideo(file: File) {
        if (!file.name.toLowerCase().endsWith('.mp4')) {
            this.error = 'Only MP4 files are supported';
            return;
        }

        if (file.size > 100 * 1024 * 1024) {
            this.error = 'File size must be less than 100MB';
            return;
        }

        await telemetryService.withSpanAsync(
            'VideoUpload.uploadVideo',
            {
                'file.name': file.name,
                'file.size': file.size,
            },
            async (span) => {
                this.uploading = true;
                this.uploadProgress = 0;
                this.error = null;
                this.success = null;

                try {
                    span?.addEvent('Starting video upload');
                    
                    this.uploadProgress = 30;
                    
                    const video = await videoService.uploadVideo(file);
                    
                    this.uploadProgress = 100;

                    if (video) {
                        this.success = `Successfully uploaded: ${video.displayName}`;
                        span?.setAttribute('video.id', video.id);
                        span?.setAttribute('upload.success', true);

                        this.dispatchEvent(new CustomEvent('video-uploaded', {
                            detail: { video },
                            bubbles: true,
                            composed: true,
                        }));

                        setTimeout(() => {
                            this.success = null;
                        }, 3000);
                    }
                } catch (err: any) {
                    this.error = err.message || 'Upload failed';
                    span?.setAttribute('error', true);
                    console.error('Upload error:', err);
                } finally {
                    this.uploading = false;
                    setTimeout(() => {
                        this.uploadProgress = 0;
                    }, 1000);
                }
            }
        );
    }

    render() {
        return html`
            <div 
                class="upload-container ${this.uploading ? 'uploading' : ''}"
                @drop=${this.handleDrop}
                @dragover=${this.handleDragOver}
                @dragleave=${this.handleDragLeave}
                data-testid="video-upload-container"
            >
                <input 
                    type="file" 
                    id="video-file-input" 
                    accept=".mp4"
                    @change=${this.handleFileSelect}
                    ?disabled=${this.uploading}
                />
                <label for="video-file-input">
                    <button 
                        class="upload-button"
                        @click=${() => this.shadowRoot?.getElementById('video-file-input')?.click()}
                        ?disabled=${this.uploading}
                        data-testid="upload-button"
                    >
                        ${this.uploading ? 'Uploading...' : 'Choose MP4 Video'}
                    </button>
                </label>
                <div class="upload-info">
                    or drag and drop an MP4 file here<br/>
                    <small>Maximum size: 100MB</small>
                </div>

                ${this.uploading ? html`
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${this.uploadProgress}%"></div>
                    </div>
                ` : ''}

                ${this.error ? html`
                    <div class="message error" data-testid="upload-error">
                        ${this.error}
                    </div>
                ` : ''}

                ${this.success ? html`
                    <div class="message success" data-testid="upload-success">
                        ${this.success}
                    </div>
                ` : ''}
            </div>
        `;
    }
}

