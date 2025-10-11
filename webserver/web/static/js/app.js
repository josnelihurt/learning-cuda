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
const resolutionSelect = document.getElementById('resolutionSelect');
const cameraStatus = document.getElementById('cameraStatus');
const toggleGPU = document.getElementById('toggleGPU');
const toggleCPU = document.getElementById('toggleCPU');
const filterGrayscale = document.getElementById('filterGrayscale');
const grayscaleParams = document.getElementById('grayscaleParams');
const grayscaleTypeSelect = document.getElementById('grayscaleTypeSelect');

// Application state
let selectedAccelerator = 'gpu'; // default to GPU

// Listen for resolution changes
if (resolutionSelect) {
    resolutionSelect.addEventListener('change', updateResolution);
}

// Accelerator toggle functions
function setAccelerator(type) {
    selectedAccelerator = type;
    
    if (type === 'gpu') {
        toggleGPU.classList.add('active');
        toggleCPU.classList.remove('active');
    } else {
        toggleCPU.classList.add('active');
        toggleGPU.classList.remove('active');
    }
    
    // Apply filter with new accelerator
    if (currentState === AppState.STATIC) {
        applyFilter();
    }
}

// Update filters and show/hide parameters
function updateFilters() {
    const isGrayscaleChecked = filterGrayscale.checked;
    
    // Show/hide grayscale parameters
    if (isGrayscaleChecked) {
        grayscaleParams.style.display = 'block';
    } else {
        grayscaleParams.style.display = 'none';
    }
    
    // Apply filters if in static mode
    if (currentState === AppState.STATIC) {
        applyFilter();
    }
}

// Get currently selected filters as array
function getSelectedFilters() {
    const filters = [];
    
    if (filterGrayscale.checked) {
        filters.push('grayscale');
    }
    
    // If no filters selected, default to none
    if (filters.length === 0) {
        filters.push('none');
    }
    
    return filters;
}

// Get current grayscale type
function getGrayscaleType() {
    return grayscaleTypeSelect.value;
}

// Switch between Lena and Webcam input
async function switchInputSource() {
    const source = inputSelect.value;
    
    if (source === 'webcam') {
        // Switch to streaming mode
        currentState = AppState.STREAMING;
        cameraStatus.style.display = 'block';
        heroImage.style.display = 'block';
        cameraPreview.style.display = 'none'; // Hide raw video, show processed
        // Allow filter changes during streaming
        
        // Start camera automatically
        await startCamera();
    } else {
        // Switch to static mode
        currentState = AppState.STATIC;
        stopCamera();
        cameraStatus.style.display = 'none';
        heroImage.style.display = 'block';
        cameraPreview.style.display = 'none';
    }
}

// Apply filter in static mode (AJAX without reload)
async function applyFilter() {
    if (currentState === AppState.STREAMING) return;
    
    const filters = getSelectedFilters();
    const filterParam = filters.includes('grayscale') ? 'grayscale' : 'none';
    const grayscaleType = getGrayscaleType();
    
    // Show loading state
    heroImage.style.opacity = '0.5';
    heroImage.style.transition = 'opacity 0.3s ease';
    
    try {
        // Fetch new image with selected filters and parameters
        const url = `/?filter=${filterParam}&accelerator=${selectedAccelerator}&grayscale_type=${grayscaleType}`;
        const response = await fetch(url);
        const html = await response.text();
        
        // Extract base64 image from response
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const newImage = doc.querySelector('#heroImage');
        
        if (newImage) {
            heroImage.src = newImage.src;
        }
        
        // Update URL without reload
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('filter', filterParam);
        newUrl.searchParams.set('accelerator', selectedAccelerator);
        newUrl.searchParams.set('grayscale_type', grayscaleType);
        window.history.pushState({}, '', newUrl);
        
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

// Processing resolution (adjustable via UI)
let PROCESS_WIDTH = 640;
let PROCESS_HEIGHT = 480;
const JPEG_QUALITY = 0.7; // 0.0 - 1.0 (lower = faster, smaller)

// Resolution presets
const RESOLUTIONS = {
    quarter: { width: 160, height: 120 },
    half: { width: 320, height: 240 },
    full: { width: 640, height: 480 }
};

function updateResolution() {
    const resolution = resolutionSelect.value;
    const preset = RESOLUTIONS[resolution];
    
    PROCESS_WIDTH = preset.width;
    PROCESS_HEIGHT = preset.height;
    
    console.log(`Resolution changed to: ${PROCESS_WIDTH}x${PROCESS_HEIGHT}`);
    
    // If streaming, restart with new resolution
    if (frameInterval) {
        stopFrameCapture();
        startFrameCapture();
    }
}

function startFrameCapture() {
    const canvas = captureCanvas;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    // Set canvas to processing resolution
    canvas.width = PROCESS_WIDTH;
    canvas.height = PROCESS_HEIGHT;
    
    console.log(`Starting capture at ${PROCESS_WIDTH}x${PROCESS_HEIGHT}`);
    
    // Capture and send frames at specified FPS
    frameInterval = setInterval(() => {
        if (!cameraPreview.videoWidth) return; // Video not ready
        if (isProcessing) return; // Skip frame if still processing previous
        
        frameCount++;
        
        // Draw scaled video frame to canvas
        ctx.drawImage(cameraPreview, 0, 0, PROCESS_WIDTH, PROCESS_HEIGHT);
        
        // Convert to JPEG for faster encoding and smaller size
        const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
        const base64data = dataUrl.split(',')[1];
        
        sendFrame(base64data, canvas.width, canvas.height);
        
    }, 1000 / FPS);
}

function stopFrameCapture() {
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
}

function sendFrame(base64Data, width, height) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    
    isProcessing = true;
    const filters = getSelectedFilters();
    const grayscaleType = getGrayscaleType();
    
    const message = {
        type: 'frame',
        filters: filters,
        accelerator: selectedAccelerator,
        grayscale_type: grayscaleType,
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
        cameraStatus.textContent = `✓ Camera active - ${frameCount} frames processed (${selectedAccelerator.toUpperCase()})`;
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
                
        try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'frame_result') {
                if (data.success) {
                    // Update hero image with processed frame
                    const newSrc = `data:image/png;base64,${data.image.data}`;
                    heroImage.src = newSrc;
                    heroImage.style.display = 'block';
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
