import './components/camera-preview';
import './components/toast-container';
import './components/stats-panel';
import './components/filter-panel';
import { WebSocketService } from './services/websocket-service';
import { UIService } from './services/ui-service';

console.log(`CUDA Image Processor v${__APP_VERSION__} (${__APP_BRANCH__}) - ${__BUILD_TIME__}`);

const app = {
    toastManager: null as any,
    statsManager: null as any,
    cameraManager: null as any,
    wsManager: null as any,
    filterManager: null as any,
    uiManager: null as any,
    
    currentState: 'static' as 'static' | 'streaming',
    selectedAccelerator: 'gpu',
    selectedInputSource: 'lena',
    
    async init() {
        console.log('Initializing dashboard...');
        await customElements.whenDefined('camera-preview');
        await customElements.whenDefined('toast-container');
        await customElements.whenDefined('stats-panel');
        await customElements.whenDefined('filter-panel');
        
        this.toastManager = document.querySelector('toast-container');
        this.toastManager.configure({ duration: 7000 });
        
        this.statsManager = document.querySelector('stats-panel');
        this.filterManager = document.querySelector('filter-panel');
        
        this.cameraManager = document.querySelector('camera-preview');
        if (this.cameraManager) {
            this.cameraManager.setManagers(this.statsManager, this.toastManager);
        }
        
        if (this.filterManager) {
            this.filterManager.addEventListener('filter-change', () => {
                if (this.currentState === 'static') {
                    this.uiManager.applyFilter();
                }
            });
        }
        
        this.wsManager = new WebSocketService(this.statsManager, this.cameraManager, this.toastManager);
        this.uiManager = new UIService(this.statsManager, this.cameraManager, this.filterManager, this.toastManager, this.wsManager);
        
        this.wsManager.connect();
        
        this.wsManager.onFrameResult((data) => {
            const heroImage = document.getElementById('heroImage');
            if (heroImage) {
                const newSrc = `data:image/png;base64,${data.image.data}`;
                (heroImage as HTMLImageElement).src = newSrc;
                (heroImage as HTMLElement).style.display = 'block';
            }
        });
        
        this.statsManager.reset();
        
        console.log('Dashboard initialized');
    }
};

(window as any).app = app;

(window as any).setInputSource = function(source: string) {
    app.selectedInputSource = app.uiManager.setInputSource(source);
    
    if (source === 'webcam') {
        switchToStreaming();
    } else {
        switchToStatic();
    }
};

(window as any).setAccelerator = function(type: string) {
    app.selectedAccelerator = app.uiManager.setAccelerator(type);
    
    if (app.currentState === 'static') {
        app.uiManager.applyFilter();
    }
};

async function switchToStreaming() {
    app.currentState = 'streaming';
    const heroImage = document.getElementById('heroImage');
    if (heroImage) {
        heroImage.style.display = 'block';
    }
    
    const success = await app.cameraManager.start();
    if (success) {
        app.cameraManager.startCapture((base64Data: string, width: number, height: number) => {
            const filters = app.filterManager.getSelectedFilters();
            const grayscaleType = app.filterManager.getGrayscaleType();
            app.wsManager.sendFrame(base64Data, width, height, filters, app.selectedAccelerator, grayscaleType);
        });
    } else {
        (window as any).setInputSource('lena');
    }
}

function switchToStatic() {
    app.currentState = 'static';
    app.cameraManager.stop();
    
    const heroImage = document.getElementById('heroImage');
    if (heroImage) {
        heroImage.style.display = 'block';
    }
    
    app.uiManager.applyFilter();
}

(window as any).updateResolution = function() {
    // Already handled by UIService initResolutionSelector
};

document.addEventListener('DOMContentLoaded', () => {
    app.init();
});

