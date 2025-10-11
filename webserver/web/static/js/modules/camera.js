/**
 * CameraManager - Handles camera access and frame capture
 */
class CameraManager {
    constructor(statsManager, toastManager) {
        this.statsManager = statsManager;
        this.toastManager = toastManager;
        this.stream = null;
        this.frameInterval = null;
        this.isProcessing = false;
        this.lastFrameTime = 0;
        
        // DOM elements
        this.preview = document.getElementById('cameraPreview');
        this.canvas = document.getElementById('captureCanvas');
        this.heroImage = document.getElementById('heroImage');
        
        // Settings
        this.FPS = 15;
        this.JPEG_QUALITY = 0.7;
        this.width = 640;
        this.height = 480;
    }
    
    setResolution(width, height) {
        this.width = width;
        this.height = height;
        
        if (this.frameInterval) {
            this.stopCapture();
            this.startCapture();
        }
    }
    
    async start() {
        try {
            this.statsManager.updateCameraStatus('Requesting access...', 'warning');
            
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error('Camera API not available. Use HTTPS.');
            }
            
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.preview.srcObject = this.stream;
            await new Promise(resolve => this.preview.onloadedmetadata = resolve);
            await this.preview.play();
            await new Promise(resolve => setTimeout(resolve, 500));
            
            console.log(`Camera ready: ${this.preview.videoWidth}x${this.preview.videoHeight}`);
            this.statsManager.updateCameraStatus('Active', 'success');
            
            return true;
        } catch (error) {
            console.error('Camera error:', error);
            
            let errorTitle = 'Camera Error';
            let errorMsg = '';
            
            if (error.name === 'NotAllowedError') {
                errorTitle = 'Permission Denied';
                errorMsg = 'Please allow camera access in your browser settings';
            } else if (error.name === 'NotFoundError') {
                errorTitle = 'No Camera Found';
                errorMsg = 'No camera device detected on this system';
            } else if (error.name === 'NotReadableError') {
                errorTitle = 'Camera In Use';
                errorMsg = 'Camera is being used by another application. Please close it and try again';
            } else if (error.message) {
                errorMsg = error.message;
            }
            
            // Show toast notification
            this.toastManager.error(errorTitle, errorMsg);
            
            // Update stats bar
            const shortMsg = errorTitle.replace(' Error', '').substring(0, 20);
            this.statsManager.updateCameraStatus(shortMsg, 'error');
            return false;
        }
    }
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.stopCapture();
        
        if (this.preview.srcObject) {
            this.preview.srcObject = null;
        }
        
        this.statsManager.updateCameraStatus('Inactive', 'inactive');
    }
    
    startCapture(onFrameCallback) {
        this.onFrameCallback = onFrameCallback;
        const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
        
        this.canvas.width = this.width;
        this.canvas.height = this.height;
        
        console.log(`Starting capture at ${this.width}x${this.height}`);
        
        this.frameInterval = setInterval(() => {
            if (!this.preview.videoWidth || this.isProcessing) return;
            
            this.lastFrameTime = performance.now();
            
            ctx.drawImage(this.preview, 0, 0, this.width, this.height);
            const dataUrl = this.canvas.toDataURL('image/jpeg', this.JPEG_QUALITY);
            const base64data = dataUrl.split(',')[1];
            
            if (this.onFrameCallback) {
                this.onFrameCallback(base64data, this.width, this.height, this.lastFrameTime);
            }
        }, 1000 / this.FPS);
    }
    
    stopCapture() {
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
    }
    
    setProcessing(isProcessing) {
        this.isProcessing = isProcessing;
    }
    
    getLastFrameTime() {
        return this.lastFrameTime;
    }
}

