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
    imageElement: HTMLImageElement | null;
    ws: WebSocketService | null;
    cameraPreview: CameraPreview | null;
    filters: string[];
    grayscaleType: string;
    resolution: string;
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
            padding: var(--spacing-xl);
            overflow: hidden;
        }

        .grid-container {
            display: grid;
            gap: 16px;
            width: 100%;
            height: 100%;
            grid-auto-rows: 1fr;
        }

        .grid-1 { grid-template-columns: 1fr; grid-template-rows: 1fr; }
        .grid-2 { grid-template-columns: 1fr; grid-template-rows: repeat(2, 1fr); }
        .grid-3 { grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); }
        .grid-4 { grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); }
        .grid-5 { grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(2, 1fr); }
        .grid-6 { grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(2, 1fr); }
        .grid-7 { grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); }
        .grid-8 { grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); }
        .grid-9 { grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); }
    `;

    render() {
        const gridClass = `grid-${this.sources.length}`;
        
        return html`
            <div class="grid-container ${gridClass}">
                ${this.sources.map(source => this.renderSourceCard(source))}
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
                .isSelected=${this.selectedSourceId === source.id}
                @source-selected=${() => this.selectSource(source.id)}
                @source-closed=${() => this.removeSource(source.id)}
            >
                ${source.imageElement 
                    ? source.imageElement
                    : html`<img src="${source.imagePath}" alt="${source.name}" />`
                }
            </video-source-card>
        `;
    }

    setManagers(statsManager: StatsPanel, toastManager: ToastContainer): void {
        this.statsManager = statsManager;
        this.toastManager = toastManager;
    }

    addSource(inputSource: InputSource): boolean {
        if (this.sources.length >= this.MAX_SOURCES) {
            this.toastManager?.warning('Maximum Sources', `Cannot add more than ${this.MAX_SOURCES} sources`);
            return false;
        }

        const number = this.nextNumber++;
        const uniqueId = `${inputSource.id}-${number}`;
        
        let cameraPreview: CameraPreview | null = null;
        let ws: WebSocketService | null = null;
        let imageElement: HTMLImageElement | null = null;

        if (inputSource.type === 'camera') {
            cameraPreview = document.createElement('camera-preview') as CameraPreview;
            if (this.statsManager && this.toastManager) {
                cameraPreview.setManagers(this.statsManager, this.toastManager);
            }
            
            ws = new WebSocketService(
                this.statsManager!,
                cameraPreview,
                this.toastManager!
            );
            ws.connect();

            ws.onFrameResult((data) => {
                const source = this.sources.find(s => s.number === number);
                if (source && source.imageElement && data.response) {
                    let binary = '';
                    const len = data.response.imageData.byteLength;
                    for (let i = 0; i < len; i++) {
                        binary += String.fromCharCode(data.response.imageData[i]);
                    }
                    const imageData = btoa(binary);
                    source.imageElement.src = `data:image/png;base64,${imageData}`;
                }
            });

            setTimeout(async () => {
                if (cameraPreview) {
                    const success = await cameraPreview.start();
                    if (success && ws) {
                        cameraPreview.startCapture((base64Data: string, width: number, height: number) => {
                            ws!.sendFrame(base64Data, width, height, ['none'], 'gpu', 'bt601');
                        });
                    }
                }
            }, 500);
        } else {
            imageElement = new Image();
            imageElement.crossOrigin = 'anonymous';
            
            ws = new WebSocketService(
                this.statsManager!,
                document.createElement('camera-preview') as CameraPreview,
                this.toastManager!
            );
            ws.connect();

            ws.onFrameResult((data) => {
                console.log('Frame result for source', number, 'success:', data.success);
                const source = this.sources.find(s => s.number === number);
                if (source && source.imageElement && data.response) {
                    let binary = '';
                    const len = data.response.imageData.byteLength;
                    for (let i = 0; i < len; i++) {
                        binary += String.fromCharCode(data.response.imageData[i]);
                    }
                    const imageData = btoa(binary);
                    const newSrc = `data:image/png;base64,${imageData}`;
                    console.log('Updating image for source', number);
                    source.imageElement.src = newSrc;
                    this.requestUpdate();
                }
            });

            imageElement.onload = () => {
                console.log('Image loaded for source:', inputSource.id, imageElement!.naturalWidth, 'x', imageElement!.naturalHeight);
                this.requestUpdate();
            };
            
            imageElement.src = inputSource.imagePath;
        }

        const newSource: GridSource = {
            id: uniqueId,
            number,
            name: inputSource.displayName,
            type: inputSource.type,
            imagePath: inputSource.imagePath,
            originalImageSrc: inputSource.imagePath,
            imageElement,
            ws,
            cameraPreview,
            filters: [],
            grayscaleType: 'bt601',
            resolution: 'original',
        };

        this.sources = [...this.sources, newSource];

        if (this.sources.length === 1) {
            this.selectSource(uniqueId);
        }

        console.log('Source added to grid:', uniqueId, 'Total:', this.sources.length);

        this.requestUpdate();
        return true;
    }

    removeSource(sourceId: string): void {
        const source = this.sources.find(s => s.id === sourceId);
        if (!source) return;

        if (source.ws) {
            source.ws.disconnect();
        }

        if (source.cameraPreview) {
            source.cameraPreview.stop();
        }

        this.sources = this.sources.filter(s => s.id !== sourceId);

        if (this.selectedSourceId === sourceId && this.sources.length > 0) {
            this.selectSource(this.sources[0].id);
        }

        console.log('Source removed from grid:', sourceId, 'Remaining:', this.sources.length);
    }

    selectSource(sourceId: string): void {
        const source = this.sources.find(s => s.id === sourceId);
        if (!source) return;

        this.selectedSourceId = sourceId;

        this.dispatchEvent(new CustomEvent('source-selection-changed', {
            bubbles: true,
            composed: true,
            detail: { 
                sourceId, 
                sourceNumber: source.number,
                sourceType: source.type,
                filters: source.filters,
                grayscaleType: source.grayscaleType,
                resolution: source.resolution
            }
        }));
    }

    getSelectedSource(): GridSource | null {
        return this.sources.find(s => s.id === this.selectedSourceId) || null;
    }

    getSelectedSourceIds(): Set<string> {
        return new Set(this.sources.map(s => s.id));
    }

    getSources(): GridSource[] {
        return this.sources;
    }

    async applyFilterToSelected(filters: string[], accelerator: string, grayscaleType: string, resolution: string = 'original'): Promise<void> {
        const selectedSource = this.getSelectedSource();
        if (!selectedSource || selectedSource.type !== 'static') {
            console.log('applyFilter: skipping non-static source');
            return;
        }

        selectedSource.filters = filters;
        selectedSource.grayscaleType = grayscaleType;
        selectedSource.resolution = resolution;

        console.log('Applying filter to source', selectedSource.number, ':', filters, grayscaleType, 'resolution:', resolution);

        if (!selectedSource.ws || !selectedSource.ws.isConnected()) {
            console.error('WebSocket not connected for source', selectedSource.number);
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

            console.log('Sending image:', originalWidth, 'x', originalHeight, '→', targetWidth, 'x', targetHeight);

            const response = await selectedSource.ws.sendSingleFrame(
                imageData,
                targetWidth,
                targetHeight,
                filters,
                accelerator,
                grayscaleType
            );

            if (response.success && response.response && selectedSource.imageElement) {
                console.log('Filter applied, updating image for source', selectedSource.number);
                let binary = '';
                const len = response.response.imageData.byteLength;
                for (let i = 0; i < len; i++) {
                    binary += String.fromCharCode(response.response.imageData[i]);
                }
                const processedImageData = btoa(binary);
                selectedSource.imageElement.src = `data:image/png;base64,${processedImageData}`;
                this.requestUpdate();
            }
        } catch (error) {
            console.error('Error applying filter:', error);
            this.toastManager?.error('Filter Error', 'Failed to apply filter');
        }
    }
}

declare global {
    interface HTMLElementTagNameMap {
        'video-grid': VideoGrid;
    }
}

