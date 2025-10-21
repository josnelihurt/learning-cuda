import './components/camera-preview';
import './components/toast-container';
import './components/stats-panel';
import './components/filter-panel';
import './components/sync-flags-button';
import './components/tools-dropdown';
import './components/video-grid';
import './components/source-drawer';
import './components/add-source-fab';
import './components/version-footer';
import './components/image-selector-modal';
import './components/video-selector';
import './components/video-upload';
import { streamConfigService } from './services/config-service';
import { telemetryService } from './services/telemetry-service';
import { logger } from './services/otel-logger';
import { inputSourceService } from './services/input-source-service';
import { processorCapabilitiesService } from './services/processor-capabilities-service';
import { toolsService } from './services/tools-service';
import { videoService } from './services/video-service';
import type { VideoGrid } from './components/video-grid';
import type { SourceDrawer } from './components/source-drawer';
import type { ToolsDropdown } from './components/tools-dropdown';
import type { ImageSelectorModal } from './components/image-selector-modal';

console.log(`CUDA Image Processor v${__APP_VERSION__} (${__APP_BRANCH__}) - ${__BUILD_TIME__}`);

const app = {
    toastManager: null as any,
    statsManager: null as any,
    filterManager: null as any,
    videoGrid: null as VideoGrid | null,
    sourceDrawer: null as SourceDrawer | null,
    toolsDropdown: null as ToolsDropdown | null,
    imageSelectorModal: null as ImageSelectorModal | null,
    currentSourceNumberForImageChange: null as number | null,
    
    selectedAccelerator: 'gpu',
    selectedResolution: 'original',
    
    async init() {
        logger.info('Initializing dashboard...');
        
        await telemetryService.initialize();
        await streamConfigService.initialize();
        
        logger.initialize(
            streamConfigService.getLogLevel(),
            streamConfigService.getConsoleLogging()
        );
        
        await inputSourceService.initialize();
        await processorCapabilitiesService.initialize();
        await toolsService.initialize();
        await videoService.initialize();
        
        await customElements.whenDefined('camera-preview');
        await customElements.whenDefined('toast-container');
        await customElements.whenDefined('stats-panel');
        await customElements.whenDefined('filter-panel');
        await customElements.whenDefined('video-grid');
        await customElements.whenDefined('source-drawer');
        await customElements.whenDefined('add-source-fab');
        await customElements.whenDefined('tools-dropdown');
        await customElements.whenDefined('image-selector-modal');
        await customElements.whenDefined('video-selector');
        await customElements.whenDefined('video-upload');
        
        this.toastManager = document.querySelector('toast-container');
        this.toastManager.configure({ duration: 7000 });
        
        this.statsManager = document.querySelector('stats-panel');
        this.filterManager = document.querySelector('filter-panel');
        this.videoGrid = document.querySelector('video-grid');
        this.sourceDrawer = document.querySelector('source-drawer');
        this.toolsDropdown = document.querySelector('tools-dropdown');
        this.imageSelectorModal = document.querySelector('image-selector-modal');
        
        if (this.filterManager && processorCapabilitiesService.isInitialized()) {
            this.filterManager.filters = processorCapabilitiesService.getFilters();
        }
        
        if (this.toolsDropdown && toolsService.isInitialized()) {
            this.toolsDropdown.categories = toolsService.getCategories();
        }
        
        if (this.videoGrid) {
            this.videoGrid.setManagers(this.statsManager, this.toastManager);
        }
        
        if (this.filterManager) {
            this.filterManager.addEventListener('filter-change', () => {
                if (this.videoGrid) {
                    const filters = this.filterManager.getSelectedFilters();
                    const grayscaleType = this.filterManager.getGrayscaleType();
                    this.videoGrid.applyFilterToSelected(filters, this.selectedAccelerator, grayscaleType, this.selectedResolution);
                }
            });
        }
        
        const resolutionSelect = document.getElementById('resolutionSelect') as HTMLSelectElement;
        if (resolutionSelect) {
            resolutionSelect.addEventListener('change', () => {
                this.selectedResolution = resolutionSelect.value;
                if (this.videoGrid && this.filterManager) {
                    const filters = this.filterManager.getSelectedFilters();
                    const grayscaleType = this.filterManager.getGrayscaleType();
                    this.videoGrid.applyFilterToSelected(filters, this.selectedAccelerator, grayscaleType, this.selectedResolution);
                }
            });
        }
        
        if (this.videoGrid) {
            this.videoGrid.addEventListener('source-selection-changed', ((e: CustomEvent) => {
                this.updateSelectedSourceIndicator(e.detail.sourceNumber, e.detail.sourceId);
                
                if (this.filterManager && e.detail.filters !== undefined) {
                    const filters = e.detail.filters.length > 0 ? e.detail.filters : [];
                    const grayscaleType = e.detail.grayscaleType || 'bt601';
                    this.filterManager.setFilters(filters, grayscaleType);
                }
                
                const resolutionSelect = document.getElementById('resolutionSelect') as HTMLSelectElement;
                if (resolutionSelect && e.detail.resolution) {
                    this.selectedResolution = e.detail.resolution;
                    resolutionSelect.value = e.detail.resolution;
                }
            }) as EventListener);
        }
        
        const fab = document.querySelector('add-source-fab');
        if (fab) {
            fab.addEventListener('open-drawer', () => {
                if (this.sourceDrawer && this.videoGrid) {
                    const sources = inputSourceService.getSources();
                    const selectedIds = this.videoGrid.getSelectedSourceIds();
                    this.sourceDrawer.open(sources, selectedIds);
                }
            });
        }
        
        if (this.sourceDrawer) {
            this.sourceDrawer.addEventListener('source-selected', ((e: CustomEvent) => {
                if (this.videoGrid) {
                    this.videoGrid.addSource(e.detail.source);
                }
            }) as EventListener);
        }
        
        if (this.videoGrid) {
            this.videoGrid.addEventListener('change-image-requested', (async (e: CustomEvent) => {
                this.currentSourceNumberForImageChange = e.detail.sourceNumber;
                if (this.imageSelectorModal) {
                    const images = await inputSourceService.listAvailableImages();
                    this.imageSelectorModal.open(images);
                }
            }) as EventListener);
        }
        
        if (this.imageSelectorModal) {
            this.imageSelectorModal.addEventListener('image-selected', ((e: CustomEvent) => {
                if (this.videoGrid && this.currentSourceNumberForImageChange !== null) {
                    const imagePath = e.detail.image.path;
                    this.videoGrid.changeSourceImage(this.currentSourceNumberForImageChange, imagePath);
                    this.currentSourceNumberForImageChange = null;
                }
            }) as EventListener);
        }
        
        const defaultSource = inputSourceService.getDefaultSource();
        if (defaultSource && this.videoGrid) {
            this.videoGrid.addSource(defaultSource);
        }
        
        this.statsManager.reset();
        
        logger.info('Dashboard initialized');
    },

    updateSelectedSourceIndicator(sourceNumber: number, sourceId: string) {
        const numberEl = document.getElementById('selectedSourceNumber');
        const nameEl = document.getElementById('selectedSourceName');
        
        if (numberEl) {
            numberEl.textContent = String(sourceNumber);
        }
        
        if (nameEl && this.videoGrid) {
            const source = this.videoGrid.getSources().find(s => s.id === sourceId);
            if (source) {
                nameEl.textContent = source.name;
            }
        }
    }
};

(window as any).app = app;

(window as any).setAccelerator = function(type: string) {
    app.selectedAccelerator = type;
    
    document.querySelectorAll('.segmented-control .segment').forEach(btn => {
        const element = btn as HTMLButtonElement;
        const controlSection = btn.closest('.control-section');
        const label = controlSection?.querySelector('.control-label');
        
        if (label?.textContent?.includes('Accelerator')) {
            element.classList.toggle('active', element.dataset.value === type);
        }
    });
    
    if (app.videoGrid && app.filterManager) {
        const filters = app.filterManager.getSelectedFilters();
        const grayscaleType = app.filterManager.getGrayscaleType();
        app.videoGrid.applyFilterToSelected(filters, type, grayscaleType, app.selectedResolution);
    }
};

document.addEventListener('DOMContentLoaded', () => {
    app.init();
});

window.addEventListener('beforeunload', () => {
    logger.shutdown();
});

