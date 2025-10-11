import { LitElement, html, css } from 'lit';

class CameraPreview extends LitElement {
    static properties = {
        width: { type: Number },
        height: { type: Number },
        fps: { type: Number },
        quality: { type: Number }
    };

    static styles = css`
        :host {
            display: contents;
        }
        
        video {
            position: absolute;
            opacity: 0;
            pointer-events: none;
        }
    `;

    constructor() {
        super();
        this.statsManager = null;
        this.toastManager = null;
        this.stream = null;
        this.frameInterval = null;
        this.isProcessing = false;
        this.lastFrameTime = 0;
        this.onFrameCallback = null;
        
        this.width = 640;
        this.height = 480;
        this.fps = 15;
        this.quality = 0.7;
    }

    render() {
        return html`
            <video autoplay playsinline muted></video>
        `;
    }

    firstUpdated() {
        this.videoElement = this.shadowRoot.querySelector('video');
    }

    setManagers(statsManager, toastManager) {
        this.statsManager = statsManager;
        this.toastManager = toastManager;
    }

    setResolution(width, height) {
        this.width = width;
        this.height = height;
        
        if (this.frameInterval) {
            this.stopCapture();
            this.startCapture(this.onFrameCallback);
        }
    }

    async start() {
        try {
            if (this.statsManager) {
                this.statsManager.updateCameraStatus('Requesting access...', 'warning');
            }
            
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
            
            this.videoElement.srcObject = this.stream;
            await new Promise(resolve => this.videoElement.onloadedmetadata = resolve);
            await this.videoElement.play();
            await new Promise(resolve => setTimeout(resolve, 500));
            
            console.log(`Camera ready: ${this.videoElement.videoWidth}x${this.videoElement.videoHeight}`);
            
            if (this.statsManager) {
                this.statsManager.updateCameraStatus('Active', 'success');
            }
            
            this.dispatchEvent(new CustomEvent('camera-started', {
                bubbles: true,
                composed: true
            }));
            
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
            
            if (this.toastManager) {
                this.toastManager.error(errorTitle, errorMsg);
            }
            
            if (this.statsManager) {
                const shortMsg = errorTitle.replace(' Error', '').substring(0, 20);
                this.statsManager.updateCameraStatus(shortMsg, 'error');
            }
            
            this.dispatchEvent(new CustomEvent('camera-error', {
                bubbles: true,
                composed: true,
                detail: { error, errorTitle, errorMsg }
            }));
            
            return false;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.stopCapture();
        
        if (this.videoElement && this.videoElement.srcObject) {
            this.videoElement.srcObject = null;
        }
        
        if (this.statsManager) {
            this.statsManager.updateCameraStatus('Inactive', 'inactive');
        }
    }

    startCapture(onFrameCallback) {
        const canvas = document.getElementById('captureCanvas');
        if (!canvas) return;
        
        this.onFrameCallback = onFrameCallback;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        
        canvas.width = this.width;
        canvas.height = this.height;
        
        console.log(`Starting capture at ${this.width}x${this.height}`);
        
        this.frameInterval = setInterval(() => {
            if (!this.videoElement.videoWidth || this.isProcessing) return;
            
            this.lastFrameTime = performance.now();
            
            ctx.drawImage(this.videoElement, 0, 0, this.width, this.height);
            const dataUrl = canvas.toDataURL('image/jpeg', this.quality);
            const base64data = dataUrl.split(',')[1];
            
            if (this.onFrameCallback) {
                this.onFrameCallback(base64data, this.width, this.height, this.lastFrameTime);
            }
            
            this.dispatchEvent(new CustomEvent('frame-captured', {
                bubbles: true,
                composed: true,
                detail: { base64data, width: this.width, height: this.height, timestamp: this.lastFrameTime }
            }));
        }, 1000 / this.fps);
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

customElements.define('camera-preview', CameraPreview);

