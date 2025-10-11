/**
 * CUDA Image Processor - Dashboard Application
 * Main entry point
 */

console.log(`🚀 CUDA Image Processor Dashboard v${APP_VERSION} - Loaded`);

// Application singleton
const app = {
    toastManager: null,
    statsManager: null,
    cameraManager: null,
    wsManager: null,
    filterManager: null,
    uiManager: null,
    
    currentState: 'static',
    selectedAccelerator: 'gpu',
    selectedInputSource: 'lena',
    
    async init() {
        console.log('Initializing dashboard...');
        
        // Initialize managers
        this.toastManager = new ToastManager({ duration: 7000 }); // 7 seconds, configurable
        this.statsManager = new StatsManager();
        this.cameraManager = new CameraManager(this.statsManager, this.toastManager);
        this.filterManager = new FilterManager();
        this.uiManager = new UIManager(this.statsManager, this.cameraManager, this.filterManager, this.toastManager);
        this.wsManager = new WebSocketManager(this.statsManager, this.cameraManager, this.toastManager);
        
        // Connect WebSocket
        this.wsManager.connect();
        
        // Setup WebSocket frame result handler
        this.wsManager.onFrameResult((data) => {
            const heroImage = document.getElementById('heroImage');
            const newSrc = `data:image/png;base64,${data.image.data}`;
            heroImage.src = newSrc;
            heroImage.style.display = 'block';
        });
        
        // Reset stats
        this.statsManager.reset();
        
        console.log('✅ Dashboard initialized');
    }
};

// Make app globally accessible
window.app = app;

/* ============================================
   GLOBAL FUNCTIONS (called from HTML)
   ============================================ */

function setInputSource(source) {
    app.selectedInputSource = app.uiManager.setInputSource(source);
    
    if (source === 'webcam') {
        switchToStreaming();
    } else {
        switchToStatic();
    }
}

function setAccelerator(type) {
    app.selectedAccelerator = app.uiManager.setAccelerator(type);
    
    if (app.currentState === 'static') {
        app.uiManager.applyFilter();
    }
}

async function switchToStreaming() {
    app.currentState = 'streaming';
    const heroImage = document.getElementById('heroImage');
    heroImage.style.display = 'block';
    
    const success = await app.cameraManager.start();
    if (success) {
        app.cameraManager.startCapture((base64Data, width, height, timestamp) => {
            app.wsManager.sendFrame(base64Data, width, height);
        });
    } else {
        // Revert to static on camera failure
        setInputSource('lena');
    }
}

function switchToStatic() {
    app.currentState = 'static';
    app.cameraManager.stop();
    
    const heroImage = document.getElementById('heroImage');
    heroImage.style.display = 'block';
    
    // Reload static image with current filters
    app.uiManager.applyFilter();
}

function toggleFilterCard(header) {
    app.filterManager.toggleCard(header);
}

function updateFilters() {
    app.filterManager.updateFiltersUI();
    
    if (app.currentState === 'static') {
        app.uiManager.applyFilter();
    }
}

function applyFilter() {
    if (app.currentState === 'static') {
        app.uiManager.applyFilter();
    }
}

function updateResolution() {
    // Already handled by UIManager initResolutionSelector
}

/* ============================================
   INITIALIZATION
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
    app.init();
});
