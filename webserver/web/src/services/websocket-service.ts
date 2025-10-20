import type { StatsPanel } from '../components/stats-panel';
import type { CameraPreview } from '../components/camera-preview';
import type { ToastContainer } from '../components/toast-container';
import { WebSocketFrameRequest, WebSocketFrameResponse, ProcessImageRequest, StartVideoPlaybackRequest, StopVideoPlaybackRequest } from '../gen/image_processor_service_pb';
import { FilterType, AcceleratorType, GrayscaleType, TraceContext } from '../gen/common_pb';
import { streamConfigService } from './config-service';
import { telemetryService } from './telemetry-service';
import { context, propagation } from '@opentelemetry/api';

type FrameResultCallback = (data: WebSocketFrameResponse) => void;

function uint8ArrayToBase64(bytes: Uint8Array): string {
    let binary = '';
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

// TODO: To be replaced by Connect-RPC bidirectional streaming client
// Target replacement: Use createPromiseClient with ImageProcessorService.streamProcessVideo
// Reference implementation: webserver/pkg/interfaces/connectrpc/handler.go StreamProcessVideo method
// Migration: Use @connectrpc/connect-web streaming API instead of native WebSocket
// Benefits: Type-safe, automatic reconnection, unified with other RPC calls, better error handling
// Keep during migration for backward compatibility
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

                if (data.type === 'frame_result' || data.type === 'video_frame') {
                    if (data.success) {
                        const sendTime = this.cameraManager.getLastFrameTime();
                        const processingTime = receiveTime - sendTime;

                        this.statsManager.updateProcessingStats(processingTime);

                        if (this.onFrameResultCallback) {
                            this.onFrameResultCallback(data);
                        }
                        
                        if (data.type === 'video_frame' && data.videoFrame) {
                            console.log('Video frame received:', data.videoFrame.frameNumber, 'frame_id:', data.videoFrame.frameId);
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

        const carrier: { [key: string]: string } = {};
        propagation.inject(context.active(), carrier);
        
        const traceContext = new TraceContext({
            traceparent: carrier['traceparent'] || '',
            tracestate: carrier['tracestate'] || '',
        });

        const frameRequest = new WebSocketFrameRequest({
            type: 'frame',
            request: request,
            traceContext: traceContext,
        });

        const transportFormat = streamConfigService.getTransportFormat();
        if (transportFormat === 'binary') {
            this.ws.send(frameRequest.toBinary());
        } else {
            this.ws.send(frameRequest.toJsonString());
        }
    }

    sendSingleFrame(base64Data: string, width: number, height: number, filters: string[], accelerator: string, grayscaleType: string): Promise<WebSocketFrameResponse> {
        return telemetryService.withSpanAsync('WebSocket.sendSingleFrame', {
            'image.width': width,
            'image.height': height,
            'filters': filters.join(','),
            'accelerator': accelerator,
            'grayscale_type': grayscaleType,
        }, async (span) => {
                return new Promise((resolve, reject) => {
                    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                        reject(new Error('WebSocket not connected'));
                        return;
                    }

                    span?.addEvent('Preparing WebSocket frame');
                    const originalCallback = this.onFrameResultCallback;
                    
                    this.onFrameResultCallback = (data: WebSocketFrameResponse) => {
                        this.onFrameResultCallback = originalCallback;
                        
                        if (data.success && data.response) {
                            span?.addEvent('Frame processed successfully');
                            span?.setAttribute('result.width', data.response.width);
                            span?.setAttribute('result.height', data.response.height);
                            resolve(data);
                        } else {
                            const error = new Error(data.error || 'Unknown error');
                            reject(error);
                        }
                    };

                    span?.addEvent('Sending frame via WebSocket');
                    this.sendFrame(base64Data, width, height, filters, accelerator, grayscaleType);
                });
        });
    }

    onFrameResult(callback: FrameResultCallback): void {
        this.onFrameResultCallback = callback;
    }

    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }

    disconnect(): void {
        if (this.ws) {
            this.ws.onclose = null;
            this.ws.close();
            this.ws = null;
            this.statsManager.updateWebSocketStatus('disconnected', 'Disconnected');
        }
    }

    sendStartVideo(videoId: string, filters: string[], accelerator: string, grayscaleType: string): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return;
        }

        const span = telemetryService.createSpan('sendStartVideo');
        
        try {
            const filterTypes = filters.map(f => {
                if (f === 'none') return FilterType.NONE;
                if (f === 'grayscale') return FilterType.GRAYSCALE;
                return FilterType.NONE;
            });

            const acceleratorType = accelerator === 'gpu' ? AcceleratorType.GPU : AcceleratorType.CPU;
            const grayscaleTypeEnum = grayscaleType === 'bt709' ? GrayscaleType.BT709 : GrayscaleType.BT601;

            const startVideoRequest = new StartVideoPlaybackRequest({
                videoId,
                filters: filterTypes,
                accelerator: acceleratorType,
                grayscaleType: grayscaleTypeEnum,
            });

            const frameRequest = new WebSocketFrameRequest({
                type: 'start_video',
                startVideoRequest,
            });

            const transportFormat = streamConfigService.getTransportFormat();
            let messageData: string | Uint8Array;

            if (transportFormat === 'binary') {
                messageData = frameRequest.toBinary();
            } else {
                messageData = frameRequest.toJsonString();
            }

            this.ws.send(messageData);
            console.log('Start video message sent:', videoId);
            
        } finally {
            if (span) {
                span.end();
            }
        }
    }

    sendStopVideo(videoId: string): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('WebSocket not connected');
            return;
        }

        const span = telemetryService.createSpan('sendStopVideo');
        
        try {
            const stopVideoRequest = new StopVideoPlaybackRequest({
                sessionId: videoId,
            });

            const frameRequest = new WebSocketFrameRequest({
                type: 'stop_video',
                stopVideoRequest,
            });

            const transportFormat = streamConfigService.getTransportFormat();
            let messageData: string | Uint8Array;

            if (transportFormat === 'binary') {
                messageData = frameRequest.toBinary();
            } else {
                messageData = frameRequest.toJsonString();
            }

            this.ws.send(messageData);
            console.log('Stop video message sent:', videoId);
            
        } finally {
            if (span) {
                span.end();
            }
        }
    }
}

