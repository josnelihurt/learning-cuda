import type { StatsPanel } from '../components/stats-panel';
import type { CameraPreview } from '../components/camera-preview';
import type { ToastContainer } from '../components/toast-container';

interface FrameMessage {
    type: 'frame';
    filters: string[];
    accelerator: string;
    grayscale_type: string;
    image: {
        data: string;
        width: number;
        height: number;
        channels: number;
    };
}

interface FrameResult {
    type: 'frame_result';
    success: boolean;
    error?: string;
    image: {
        data: string;
    };
}

type FrameResultCallback = (data: FrameResult) => void;

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
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

        this.ws.onopen = () => {
            this.statsManager.updateWebSocketStatus('connected', 'Connected');
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const receiveTime = performance.now();
            this.cameraManager.setProcessing(false);

            try {
                const data = JSON.parse(event.data) as FrameResult;

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

        const message: FrameMessage = {
            type: 'frame',
            filters,
            accelerator,
            grayscale_type: grayscaleType,
            image: {
                data: base64Data,
                width,
                height,
                channels: 4
            }
        };

        this.ws.send(JSON.stringify(message));
    }

    sendSingleFrame(base64Data: string, width: number, height: number, filters: string[], accelerator: string, grayscaleType: string): Promise<void> {
        return new Promise((resolve, reject) => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                reject(new Error('WebSocket not connected'));
                return;
            }

            const originalCallback = this.onFrameResultCallback;
            
            this.onFrameResultCallback = (data: FrameResult) => {
                this.onFrameResultCallback = originalCallback;
                
                if (data.success) {
                    const heroImage = document.querySelector('#heroImage') as HTMLImageElement;
                    if (heroImage) {
                        heroImage.src = `data:image/png;base64,${data.image.data}`;
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

