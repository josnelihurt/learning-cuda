/**
 * WebSocketManager - Handles WebSocket connection and messaging
 */
class WebSocketManager {
    constructor(statsManager, cameraManager, toastManager) {
        this.statsManager = statsManager;
        this.cameraManager = cameraManager;
        this.toastManager = toastManager;
        this.ws = null;
        this.reconnectTimeout = 3000;
        this.onFrameResultCallback = null;
    }
    
    connect() {
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
                const data = JSON.parse(event.data);
                
                if (data.type === 'frame_result') {
                    if (data.success) {
                        // Calculate processing time
                        const sendTime = this.cameraManager.getLastFrameTime();
                        const processingTime = receiveTime - sendTime;
                        
                        // Update stats
                        this.statsManager.updateProcessingStats(processingTime);
                        
                        // Callback for UI update
                        if (this.onFrameResultCallback) {
                            this.onFrameResultCallback(data);
                        }
                    } else {
                        console.error('Frame processing error:', data.error);
                        this.toastManager.error('Processing Error', data.error);
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
            this.statsManager.updateWebSocketStatus('error', 'Connection error');
            this.cameraManager.setProcessing(false);
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed - Reconnecting...');
            this.statsManager.updateWebSocketStatus('connecting', 'Reconnecting...');
            setTimeout(() => this.connect(), this.reconnectTimeout);
        };
    }
    
    sendFrame(base64Data, width, height) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        
        this.cameraManager.setProcessing(true);
        
        const message = {
            type: 'frame',
            filters: window.app.filterManager.getSelectedFilters(),
            accelerator: window.app.selectedAccelerator,
            grayscale_type: window.app.filterManager.getGrayscaleType(),
            image: {
                data: base64Data,
                width: width,
                height: height,
                channels: 4
            }
        };
        
        this.ws.send(JSON.stringify(message));
    }
    
    onFrameResult(callback) {
        this.onFrameResultCallback = callback;
    }
    
    isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}

