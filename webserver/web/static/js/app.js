console.log('🚀 CUDA Image Processor JS v2.1 - Loaded');

// Application State
const AppState = {
    STATIC: 'static',
    STREAMING: 'streaming'
};

let currentState = AppState.STATIC;
let ws = null;
let cameraStream = null;
let frameInterval = null;
const FPS = 15; // Frames per second for streaming

// DOM Elements
const statusBar = document.getElementById('statusBar');
const statusText = document.getElementById('statusText');
const heroImage = document.getElementById('heroImage');
const cameraPreview = document.getElementById('cameraPreview');
const captureCanvas = document.getElementById('captureCanvas');
const inputSelect = document.getElementById('inputSelect');
const filterSelect = document.getElementById('filterSelect');
const cameraStatus = document.getElementById('cameraStatus');

// Switch between Lena and Webcam input
async function switchInputSource() {
    const source = inputSelect.value;
    
    if (source === 'webcam') {
        // Switch to streaming mode
        currentState = AppState.STREAMING;
        cameraStatus.style.display = 'block';
        heroImage.style.display = 'block';
        cameraPreview.style.display = 'none'; // Hide raw video, show processed
        filterSelect.disabled = false; // Allow filter changes during streaming
        
        // Start camera automatically
        await startCamera();
    } else {
        // Switch to static mode
        currentState = AppState.STATIC;
        stopCamera();
        cameraStatus.style.display = 'none';
        heroImage.style.display = 'block';
        cameraPreview.style.display = 'none';
        filterSelect.disabled = false;
    }
}

// Apply filter in static mode (AJAX without reload)
async function applyFilter() {
    if (currentState === AppState.STREAMING) return;
    
    const filter = filterSelect.value;
    
    // Show loading state
    heroImage.style.opacity = '0.5';
    heroImage.style.transition = 'opacity 0.3s ease';
    
    try {
        // Fetch new image with selected filter
        const response = await fetch(`/?filter=${filter}`);
        const html = await response.text();
        
        // Extract base64 image from response
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const newImage = doc.querySelector('#heroImage');
        
        if (newImage) {
            heroImage.src = newImage.src;
        }
        
        // Update URL without reload
        const url = new URL(window.location);
        url.searchParams.set('filter', filter);
        window.history.pushState({}, '', url);
        
    } catch (error) {
        console.error('Error applying filter:', error);
        updateStatus('Error processing image', 'error');
    } finally {
        heroImage.style.opacity = '1';
    }
}

// Camera Management
async function startCamera() {
    try {
        cameraStatus.textContent = 'Requesting camera access...';
        cameraStatus.style.color = '#939393';
        
        // Debug: Log available APIs
        console.log('navigator.mediaDevices:', navigator.mediaDevices);
        console.log('Location:', window.location.href);
        console.log('isSecureContext:', window.isSecureContext);
        
        // Check if getUserMedia is available with better detection
        const getUserMedia = navigator.mediaDevices?.getUserMedia ||
                            navigator.getUserMedia ||
                            navigator.webkitGetUserMedia ||
                            navigator.mozGetUserMedia;
        
        if (!getUserMedia) {
            throw new Error(`Camera API not available. Protocol: ${window.location.protocol}, Host: ${window.location.hostname}. Try accessing via http://localhost:8080`);
        }
        
        // Request camera access (use modern API if available)
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
        } else {
            throw new Error('Modern camera API not available');
        }
        
        cameraPreview.srcObject = cameraStream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            cameraPreview.onloadedmetadata = resolve;
        });
        
        // CRITICAL: Wait for video to actually play and have frames
        await cameraPreview.play();
        await new Promise(resolve => setTimeout(resolve, 500)); // Wait 500ms for frames
        
        console.log(`Video ready: ${cameraPreview.videoWidth}x${cameraPreview.videoHeight}`);
        
        cameraStatus.textContent = '✓ Camera active - Processing frames at 15 FPS';
        cameraStatus.style.color = '#ffa400';
        
        // Start frame capture and streaming
        startFrameCapture();
        
    } catch (error) {
        console.error('Camera access error:', error);
        let errorMsg = '✗ Camera access failed';
        
        // Provide specific error messages
        if (error.name === 'NotAllowedError') {
            errorMsg = '✗ Camera permission denied. Please allow camera access and try again.';
        } else if (error.name === 'NotFoundError') {
            errorMsg = '✗ No camera found on this device.';
        } else if (error.name === 'NotReadableError') {
            errorMsg = '✗ Camera is already in use by another application.';
        } else if (error.message) {
            errorMsg = `✗ ${error.message}`;
        }
        
        cameraStatus.textContent = errorMsg;
        cameraStatus.style.color = '#d32f2f';
        
        // Revert to Lena mode
        inputSelect.value = 'lena';
        await switchInputSource();
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    
    if (cameraPreview.srcObject) {
        cameraPreview.srcObject = null;
    }
    
    cameraStatus.textContent = '';
    cameraStatus.style.color = '#939393';
}

// Frame Capture and Streaming
let isProcessing = false;
let frameCount = 0;

function startFrameCapture() {
    const canvas = captureCanvas;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Capture and send frames at specified FPS
    frameInterval = setInterval(() => {
        if (!cameraPreview.videoWidth) return; // Video not ready
        if (isProcessing) return; // Skip frame if still processing previous
        
        frameCount++;
        
        // Set canvas size to match video
        canvas.width = cameraPreview.videoWidth;
        canvas.height = cameraPreview.videoHeight;
        
        // Draw current video frame to canvas
        ctx.drawImage(cameraPreview, 0, 0);
        
        // Convert to base64 PNG
        const dataUrl = canvas.toDataURL('image/png');
        const base64data = dataUrl.split(',')[1];
        
        // Debug: Log capture info
        if (frameCount % 30 === 1) {
            console.log(`Capturing frame ${frameCount}: ${canvas.width}x${canvas.height}, data size: ${base64data.length} chars`);
        }
        
        sendFrame(base64data, canvas.width, canvas.height);
        
    }, 1000 / FPS);
}

function sendFrame(base64Data, width, height) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    
    isProcessing = true;
    const filter = filterSelect.value;
    
    const message = {
        type: 'frame',
        filter: filter,
        image: {
            data: base64Data,
            width: width,
            height: height,
            channels: 4 // RGBA
        }
    };
    
    ws.send(JSON.stringify(message));
    
    // Update status with frame count
    if (frameCount % 30 === 0) { // Update every 30 frames (~2 seconds)
        cameraStatus.textContent = `✓ Camera active - ${frameCount} frames processed`;
    }
}

// WebSocket connection management
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = function() {
        updateStatus('Connected - Real-time processing ready', 'connected');
        console.log('WebSocket connected');
    };

    ws.onmessage = function(event) {
        isProcessing = false; // Always reset processing flag
        
        console.log('WebSocket received:', event.data.substring(0, 100) + '...');
        
        try {
            const data = JSON.parse(event.data);
            console.log('Parsed data type:', data.type, 'success:', data.success);
            
            if (data.type === 'frame_result') {
                if (data.success) {
                    // Update hero image with processed frame
                    const newSrc = `data:image/png;base64,${data.image.data}`;
                    heroImage.src = newSrc;
                    heroImage.style.display = 'block';
                    
                    console.log(`✓ Frame ${frameCount} - Image updated, size: ${data.image.data.length} bytes`);
                } else {
                    console.error('Frame processing error:', data.error);
                    cameraStatus.textContent = `⚠ Error: ${data.error}`;
                    cameraStatus.style.color = '#ff9100';
                }
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateStatus('Connection error - Retrying...', 'error');
        isProcessing = false; // Reset processing on error
    };

    ws.onclose = function() {
        console.log('WebSocket closed - Reconnecting...');
        updateStatus('Reconnecting...', '');
        setTimeout(connectWebSocket, 3000);
    };
}

// Status bar helper
function updateStatus(message, className) {
    statusText.textContent = message;
    statusBar.classList.remove('connected', 'error');
    if (className) {
        statusBar.classList.add(className);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    connectWebSocket();
});
