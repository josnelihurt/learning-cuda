import type { StatsPanel } from '../components/stats-panel';
import type { CameraPreview } from '../components/camera-preview';
import type { ToastContainer } from '../components/toast-container';
import { WebSocketFrameRequest, WebSocketFrameResponse, ProcessImageRequest, FilterType, AcceleratorType, GrayscaleType } from '../gen/image_processing_pb';
import { streamConfigService } from './config-service';

type FrameResultCallback = (data: WebSocketFrameResponse) => void;

function uint8ArrayToBase64(bytes: Uint8Array): string {
    let binary = '';
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

export class WebSocketService {
    private ws: WebSocket | null = null;
    private reconnectTimeout = 3000;
    private onFrameResultCallback: FrameResultCallback | null = null;

    constructor(
        private statsManager: StatsPanel,
        private cameraManager: CameraPreview,
        private toastManager: ToastContainer
    ) {}

    connect(): void {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const endpoint = streamConfigService.getWebSocketEndpoint();
        this.ws = new WebSocket(`${protocol}//${window.location.host}${endpoint}`);

        this.ws.onopen = () => {
            this.statsManager.updateWebSocketStatus('connected', 'Connected');
            console.log('WebSocket connected');
        };

        this.ws.onmessage = async (event) => {
            const receiveTime = performance.now();
            this.cameraManager.setProcessing(false);

            try {
                let data: WebSocketFrameResponse;
                const transportFormat = streamConfigService.getTransportFormat();

                if (transportFormat === 'binary') {
                    const buffer = await event.data.arrayBuffer();
                    data = WebSocketFrameResponse.fromBinary(new Uint8Array(buffer));
                } else {
                    data = WebSocketFrameResponse.fromJsonString(event.data);
                }

                if (data.type === 'frame_result') {
                    if (data.success) {
                        const sendTime = this.cameraManager.getLastFrameTime();
                        const processingTime = receiveTime - sendTime;

                        this.statsManager.updateProcessingStats(processingTime);

                        if (this.onFrameResultCallback) {
                            this.onFrameResultCallback(data);
                        }
                    } else {
                        console.error('Frame processing error:', data.error);
                        this.toastManager.error('Processing Error', data.error || 'Unknown error');
                        this.statsManager.updateCameraStatus('Processing failed', 'error');
                    }
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.toastManager.warning('WebSocket Error', 'Connection error, attempting to reconnect...');
            this.statsManager.updateWebSocketStatus('disconnected', 'Connection error');
            this.cameraManager.setProcessing(false);
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed - Reconnecting...');
            this.statsManager.updateWebSocketStatus('connecting', 'Reconnecting...');
            setTimeout(() => this.connect(), this.reconnectTimeout);
        };
    }

    sendFrame(base64Data: string, width: number, height: number, filters: string[], accelerator: string, grayscaleType: string): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        this.cameraManager.setProcessing(true);

        const protoFilters = filters.map(f => {
            switch (f) {
                case 'none': return FilterType.NONE;
                case 'grayscale': return FilterType.GRAYSCALE;
                default: return FilterType.NONE;
            }
        });

        const protoAccelerator = accelerator === 'cpu' 
            ? AcceleratorType.CPU 
            : AcceleratorType.GPU;

        let protoGrayscaleType: GrayscaleType;
        switch (grayscaleType) {
            case 'bt601': protoGrayscaleType = GrayscaleType.BT601; break;
            case 'bt709': protoGrayscaleType = GrayscaleType.BT709; break;
            case 'average': protoGrayscaleType = GrayscaleType.AVERAGE; break;
            case 'lightness': protoGrayscaleType = GrayscaleType.LIGHTNESS; break;
            case 'luminosity': protoGrayscaleType = GrayscaleType.LUMINOSITY; break;
            default: protoGrayscaleType = GrayscaleType.BT601;
        }

        const imageDataB64 = base64Data.replace(/^data:image\/(png|jpeg);base64,/, '');
        const imageBytes = Uint8Array.from(atob(imageDataB64), c => c.charCodeAt(0));

        const request = new ProcessImageRequest({
            imageData: imageBytes,
            width,
            height,
            channels: 4,
            filters: protoFilters,
            accelerator: protoAccelerator,
            grayscaleType: protoGrayscaleType,
        });

        const frameRequest = new WebSocketFrameRequest({
            type: 'frame',
            request: request,
        });

        const transportFormat = streamConfigService.getTransportFormat();
        if (transportFormat === 'binary') {
            this.ws.send(frameRequest.toBinary());
        } else {
            this.ws.send(frameRequest.toJsonString());
        }
    }

    sendSingleFrame(base64Data: string, width: number, height: number, filters: string[], accelerator: string, grayscaleType: string): Promise<void> {
        return new Promise((resolve, reject) => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                reject(new Error('WebSocket not connected'));
                return;
            }

            const originalCallback = this.onFrameResultCallback;
            
            this.onFrameResultCallback = (data: WebSocketFrameResponse) => {
                this.onFrameResultCallback = originalCallback;
                
                if (data.success && data.response) {
                    const heroImage = document.querySelector('#heroImage') as HTMLImageElement;
                    if (heroImage) {
                        const imageData = uint8ArrayToBase64(data.response.imageData);
                        heroImage.src = `data:image/png;base64,${imageData}`;
                    }
                    resolve();
                } else {
                    reject(new Error(data.error || 'Unknown error'));
                }
            };

            this.sendFrame(base64Data, width, height, filters, accelerator, grayscaleType);
        });
    }

    onFrameResult(callback: FrameResultCallback): void {
        this.onFrameResultCallback = callback;
    }

    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }
}

