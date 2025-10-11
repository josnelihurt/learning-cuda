console.log('🚀 CUDA Image Processor Dashboard v3.0 - Loaded');

// Application State
const AppState = {
    STATIC: 'static',
    STREAMING: 'streaming'
};

let currentState = AppState.STATIC;
let ws = null;
let cameraStream = null;
let frameInterval = null;
const FPS = 15;

// Stats tracking
let frameCount = 0;
let fpsHistory = [];
let lastFrameTime = 0;
let processingTimes = [];

// Filter order (for drag & drop)
let filterOrder = ['grayscale'];

// DOM Elements
const heroImage = document.getElementById('heroImage');
const cameraPreview = document.getElementById('cameraPreview');
const captureCanvas = document.getElementById('captureCanvas');
const resolutionSelect = document.getElementById('resolutionSelect');
const filterGrayscale = document.getElementById('filterGrayscale');
const infoBtn = document.getElementById('infoBtn');
const infoTooltip = document.getElementById('infoTooltip');

// Stats DOM elements
const statFPS = document.getElementById('statFPS');
const statTime = document.getElementById('statTime');
const statFrames = document.getElementById('statFrames');
const statCamera = document.getElementById('statCamera');
const wsIndicator = document.getElementById('wsIndicator');
const wsStatus = document.getElementById('wsStatus');

// Application state
let selectedInputSource = 'lena';
let selectedAccelerator = 'gpu';

// Resolution presets
const RESOLUTIONS = {
    quarter: { width: 160, height: 120 },
    half: { width: 320, height: 240 },
    full: { width: 640, height: 480 }
};

let PROCESS_WIDTH = 640;
let PROCESS_HEIGHT = 480;
const JPEG_QUALITY = 0.7;

/* ============================================
   INFO TOOLTIP
   ============================================ */

if (infoBtn && infoTooltip) {
    infoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        infoTooltip.style.display = infoTooltip.style.display === 'none' ? 'block' : 'none';
    });
    
    document.addEventListener('click', (e) => {
        if (!infoBtn.contains(e.target) && !infoTooltip.contains(e.target)) {
            infoTooltip.style.display = 'none';
        }
    });
}

/* ============================================
   SEGMENTED CONTROLS
   ============================================ */

function setInputSource(source) {
    selectedInputSource = source;
    
    // Update UI
    document.querySelectorAll('.segmented-control .segment').forEach(btn => {
        if (btn.closest('.control-section').querySelector('.control-label').textContent.includes('Input')) {
            btn.classList.toggle('active', btn.dataset.value === source);
        }
    });
    
    // Switch mode
    if (source === 'webcam') {
        switchToStreaming();
    } else {
        switchToStatic();
    }
}

function setAccelerator(type) {
    selectedAccelerator = type;
    
    // Update UI
    document.querySelectorAll('.segmented-control .segment').forEach(btn => {
        if (btn.closest('.control-section').querySelector('.control-label').textContent.includes('Accelerator')) {
            btn.classList.toggle('active', btn.dataset.value === type);
        }
    });
    
    // Apply filter if in static mode
    if (currentState === AppState.STATIC) {
        applyFilter();
    }
}

/* ============================================
   INPUT SOURCE SWITCHING
   ============================================ */

async function switchToStreaming() {
    currentState = AppState.STREAMING;
    heroImage.style.display = 'block';
    cameraPreview.style.display = 'none';
    
    updateCameraStatus('Starting...', 'warning');
    await startCamera();
}

async function switchToStatic() {
    currentState = AppState.STATIC;
    stopCamera();
    heroImage.style.display = 'block';
    cameraPreview.style.display = 'none';
    updateCameraStatus('Inactive', 'inactive');
    
    // Reload static image with current filters
    applyFilter();
}

/* ============================================
   FILTER MANAGEMENT
   ============================================ */

function toggleFilterCard(header) {
    const card = header.parentElement;
    const body = card.querySelector('.filter-body');
    const chevron = header.querySelector('.chevron');
    
    if (card.classList.contains('disabled')) return;
    
    const isExpanded = body.classList.contains('expanded');
    
    if (isExpanded) {
        body.classList.remove('expanded');
        header.classList.add('collapsed');
    } else {
        body.classList.add('expanded');
        header.classList.remove('collapsed');
    }
}

function updateFilters() {
    // Apply filters if in static mode
    if (currentState === AppState.STATIC) {
        applyFilter();
    }
}

function getSelectedFilters() {
    const filters = [];
    
    if (filterGrayscale && filterGrayscale.checked) {
        filters.push('grayscale');
    }
    
    return filters.length > 0 ? filters : ['none'];
}

function getGrayscaleType() {
    const selected = document.querySelector('input[name="grayscale-algo"]:checked');
    return selected ? selected.value : 'bt601';
}

/* ============================================
   FILTER APPLICATION
   ============================================ */

async function applyFilter() {
    if (currentState === AppState.STREAMING) return;
    
    const filters = getSelectedFilters();
    const filterParam = filters.includes('grayscale') ? 'grayscale' : 'none';
    const grayscaleType = getGrayscaleType();
    
    heroImage.classList.add('loading');
    
    try {
        const url = `/?filter=${filterParam}&accelerator=${selectedAccelerator}&grayscale_type=${grayscaleType}`;
        const response = await fetch(url);
        const html = await response.text();
        
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const newImage = doc.querySelector('#heroImage');
        
        if (newImage) {
            heroImage.src = newImage.src;
        }
        
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('filter', filterParam);
        newUrl.searchParams.set('accelerator', selectedAccelerator);
        newUrl.searchParams.set('grayscale_type', grayscaleType);
        window.history.pushState({}, '', newUrl);
        
    } catch (error) {
        console.error('Error applying filter:', error);
        updateWSStatus('error', 'Error processing image');
    } finally {
        heroImage.classList.remove('loading');
    }
}

/* ============================================
   CAMERA MANAGEMENT
   ============================================ */

async function startCamera() {
    try {
        updateCameraStatus('Requesting access...', 'warning');
        
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('Camera API not available. Use HTTPS: https://localhost:8443');
        }
        
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });
        
        cameraPreview.srcObject = cameraStream;
        
        await new Promise((resolve) => {
            cameraPreview.onloadedmetadata = resolve;
        });
        
        await cameraPreview.play();
        await new Promise(resolve => setTimeout(resolve, 500));
        
        console.log(`Camera ready: ${cameraPreview.videoWidth}x${cameraPreview.videoHeight}`);
        
        updateCameraStatus('Active', 'success');
        startFrameCapture();
        
    } catch (error) {
        console.error('Camera access error:', error);
        let errorMsg = 'Failed';
        
        if (error.name === 'NotAllowedError') {
            errorMsg = 'Permission denied';
        } else if (error.name === 'NotFoundError') {
            errorMsg = 'No camera found';
        } else if (error.name === 'NotReadableError') {
            errorMsg = 'Camera in use';
        } else if (error.message) {
            errorMsg = error.message.substring(0, 30);
        }
        
        updateCameraStatus(errorMsg, 'error');
        setInputSource('lena');
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
}

function updateResolution() {
    const resolution = resolutionSelect.value;
    const preset = RESOLUTIONS[resolution];
    
    PROCESS_WIDTH = preset.width;
    PROCESS_HEIGHT = preset.height;
    
    console.log(`Resolution: ${PROCESS_WIDTH}x${PROCESS_HEIGHT}`);
    
    if (frameInterval) {
        stopFrameCapture();
        startFrameCapture();
    }
}

/* ============================================
   FRAME CAPTURE & STREAMING
   ============================================ */

let isProcessing = false;

function startFrameCapture() {
    const canvas = captureCanvas;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    canvas.width = PROCESS_WIDTH;
    canvas.height = PROCESS_HEIGHT;
    
    console.log(`Starting capture at ${PROCESS_WIDTH}x${PROCESS_HEIGHT}`);
    
    frameInterval = setInterval(() => {
        if (!cameraPreview.videoWidth) return;
        if (isProcessing) return;
        
        frameCount++;
        
        ctx.drawImage(cameraPreview, 0, 0, PROCESS_WIDTH, PROCESS_HEIGHT);
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
    const sendTime = performance.now();
    
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
            channels: 4
        },
        timestamp: sendTime
    };
    
    ws.send(JSON.stringify(message));
}

/* ============================================
   WEBSOCKET
   ============================================ */

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = function() {
        updateWSStatus('connected', 'Connected');
        console.log('WebSocket connected');
    };

    ws.onmessage = function(event) {
        const receiveTime = performance.now();
        isProcessing = false;
                
        try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'frame_result') {
                if (data.success) {
                    const newSrc = `data:image/png;base64,${data.image.data}`;
                    heroImage.src = newSrc;
                    heroImage.style.display = 'block';
                    
                    // Calculate processing time
                    if (data.timestamp) {
                        const processingTime = receiveTime - data.timestamp;
                        updateStats(processingTime);
                    }
                } else {
                    console.error('Frame processing error:', data.error);
                    updateCameraStatus(`Error: ${data.error}`, 'error');
                }
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateWSStatus('error', 'Connection error');
        isProcessing = false;
    };

    ws.onclose = function() {
        console.log('WebSocket closed - Reconnecting...');
        updateWSStatus('connecting', 'Reconnecting...');
        setTimeout(connectWebSocket, 3000);
    };
}

/* ============================================
   STATS TRACKING
   ============================================ */

function updateStats(processingTime) {
    // Update frame count
    statFrames.textContent = frameCount;
    
    // Calculate FPS (sliding window of last 10 frames)
    const now = performance.now();
    if (lastFrameTime > 0) {
        const frameDuration = now - lastFrameTime;
        const instantFPS = 1000 / frameDuration;
        
        fpsHistory.push(instantFPS);
        if (fpsHistory.length > 10) fpsHistory.shift();
        
        const avgFPS = fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length;
        statFPS.textContent = avgFPS.toFixed(1);
    }
    lastFrameTime = now;
    
    // Update processing time (average of last 10)
    processingTimes.push(processingTime);
    if (processingTimes.length > 10) processingTimes.shift();
    
    const avgTime = processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length;
    statTime.textContent = avgTime.toFixed(0) + 'ms';
}

function updateCameraStatus(status, type) {
    statCamera.textContent = status;
    statCamera.className = '';
    
    if (type === 'success') {
        statCamera.style.color = 'var(--success)';
    } else if (type === 'error') {
        statCamera.style.color = 'var(--error)';
    } else if (type === 'warning') {
        statCamera.style.color = 'var(--warning)';
    } else {
        statCamera.style.color = '#939393';
    }
}

function updateWSStatus(status, text) {
    wsIndicator.className = 'ws-indicator ' + status;
    wsStatus.textContent = text;
}

/* ============================================
   DRAG & DROP - Filter Reordering
   ============================================ */

let draggedElement = null;
let draggedIndex = null;

function initDragAndDrop() {
    const filtersList = document.getElementById('filtersList');
    const cards = filtersList.querySelectorAll('.filter-card:not(.disabled)');
    
    cards.forEach((card, index) => {
        card.addEventListener('dragstart', handleDragStart);
        card.addEventListener('dragend', handleDragEnd);
        card.addEventListener('dragover', handleDragOver);
        card.addEventListener('drop', handleDrop);
        card.addEventListener('dragenter', handleDragEnter);
        card.addEventListener('dragleave', handleDragLeave);
    });
}

function handleDragStart(e) {
    draggedElement = this;
    draggedIndex = Array.from(this.parentNode.children).indexOf(this);
    this.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
}

function handleDragEnd(e) {
    this.classList.remove('dragging');
    
    // Remove drag-over class from all cards
    document.querySelectorAll('.filter-card').forEach(card => {
        card.classList.remove('drag-over');
    });
}

function handleDragOver(e) {
    if (e.preventDefault) {
        e.preventDefault();
    }
    e.dataTransfer.dropEffect = 'move';
    return false;
}

function handleDragEnter(e) {
    if (this !== draggedElement && !this.classList.contains('disabled')) {
        this.classList.add('drag-over');
    }
}

function handleDragLeave(e) {
    this.classList.remove('drag-over');
}

function handleDrop(e) {
    if (e.stopPropagation) {
        e.stopPropagation();
    }
    
    if (draggedElement !== this && !this.classList.contains('disabled')) {
        const dropIndex = Array.from(this.parentNode.children).indexOf(this);
        
        // Reorder DOM elements
        if (draggedIndex < dropIndex) {
            this.parentNode.insertBefore(draggedElement, this.nextSibling);
        } else {
            this.parentNode.insertBefore(draggedElement, this);
        }
        
        // Update filter order
        updateFilterOrder();
        console.log('Filter order:', filterOrder);
    }
    
    return false;
}

function updateFilterOrder() {
    const filtersList = document.getElementById('filtersList');
    const cards = filtersList.querySelectorAll('.filter-card:not(.disabled)');
    filterOrder = Array.from(cards).map(card => card.dataset.filter);
}

/* ============================================
   APPLY FILTER
   ============================================ */

async function applyFilter() {
    if (currentState === AppState.STREAMING) return;
    
    const filters = getSelectedFilters();
    const filterParam = filters.includes('grayscale') ? 'grayscale' : 'none';
    const grayscaleType = getGrayscaleType();
    
    const startTime = performance.now();
    heroImage.classList.add('loading');
    
    try {
        const url = `/?filter=${filterParam}&accelerator=${selectedAccelerator}&grayscale_type=${grayscaleType}`;
        const response = await fetch(url);
        const html = await response.text();
        
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const newImage = doc.querySelector('#heroImage');
        
        if (newImage) {
            heroImage.src = newImage.src;
        }
        
        const processingTime = performance.now() - startTime;
        statTime.textContent = processingTime.toFixed(0) + 'ms';
        
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('filter', filterParam);
        newUrl.searchParams.set('accelerator', selectedAccelerator);
        newUrl.searchParams.set('grayscale_type', grayscaleType);
        window.history.pushState({}, '', newUrl);
        
    } catch (error) {
        console.error('Error applying filter:', error);
        updateWSStatus('error', 'Error processing');
    } finally {
        heroImage.classList.remove('loading');
    }
}

/* ============================================
   CAMERA FUNCTIONALITY
   ============================================ */

async function startCamera() {
    try {
        const getUserMedia = navigator.mediaDevices?.getUserMedia;
        
        if (!getUserMedia) {
            throw new Error('Camera API not available. Use HTTPS.');
        }
        
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
        
        await new Promise((resolve) => {
            cameraPreview.onloadedmetadata = resolve;
        });
        
        await cameraPreview.play();
        await new Promise(resolve => setTimeout(resolve, 500));
        
        console.log(`Camera ready: ${cameraPreview.videoWidth}x${cameraPreview.videoHeight}`);
        
        updateCameraStatus('Active', 'success');
        startFrameCapture();
        
    } catch (error) {
        console.error('Camera error:', error);
        let errorMsg = 'Failed';
        
        if (error.name === 'NotAllowedError') {
            errorMsg = 'Permission denied';
        } else if (error.name === 'NotFoundError') {
            errorMsg = 'No camera found';
        } else if (error.name === 'NotReadableError') {
            errorMsg = 'Camera in use';
        } else if (error.message) {
            errorMsg = error.message.substring(0, 25);
        }
        
        updateCameraStatus(errorMsg, 'error');
        setInputSource('lena');
    }
}

/* ============================================
   INITIALIZATION
   ============================================ */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing CUDA Image Processor Dashboard...');
    
    // Connect WebSocket
    connectWebSocket();
    
    // Initialize drag and drop
    initDragAndDrop();
    
    // Reset stats
    statFPS.textContent = '--';
    statTime.textContent = '--ms';
    statFrames.textContent = '0';
    updateCameraStatus('Inactive', 'inactive');
    
    console.log('✅ Dashboard initialized');
});
