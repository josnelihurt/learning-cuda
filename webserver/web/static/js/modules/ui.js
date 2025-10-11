/**
 * UIManager - Handles UI controls and interactions
 */
class UIManager {
    constructor(statsManager, cameraManager, filterManager, toastManager) {
        this.statsManager = statsManager;
        this.cameraManager = cameraManager;
        this.filterManager = filterManager;
        this.toastManager = toastManager;
        
        this.selectedInputSource = 'lena';
        this.selectedAccelerator = 'gpu';
        this.currentState = 'static'; // 'static' or 'streaming'
        
        // DOM elements
        this.heroImage = document.getElementById('heroImage');
        this.resolutionSelect = document.getElementById('resolutionSelect');
        this.infoBtn = document.getElementById('infoBtn');
        this.infoTooltip = document.getElementById('infoTooltip');
        
        this.initInfoTooltip();
        this.initResolutionSelector();
    }
    
    initInfoTooltip() {
        if (this.infoBtn && this.infoTooltip) {
            this.infoBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.infoTooltip.style.display = 
                    this.infoTooltip.style.display === 'none' ? 'block' : 'none';
            });
            
            document.addEventListener('click', (e) => {
                if (!this.infoBtn.contains(e.target) && !this.infoTooltip.contains(e.target)) {
                    this.infoTooltip.style.display = 'none';
                }
            });
        }
    }
    
    initResolutionSelector() {
        if (this.resolutionSelect) {
            this.resolutionSelect.addEventListener('change', () => {
                const resolution = this.resolutionSelect.value;
                const presets = {
                    quarter: { width: 160, height: 120 },
                    half: { width: 320, height: 240 },
                    full: { width: 640, height: 480 }
                };
                
                const preset = presets[resolution];
                this.cameraManager.setResolution(preset.width, preset.height);
            });
        }
    }
    
    setInputSource(source) {
        this.selectedInputSource = source;
        
        // Update UI
        document.querySelectorAll('.segmented-control .segment').forEach(btn => {
            if (btn.closest('.control-section').querySelector('.control-label').textContent.includes('Input')) {
                btn.classList.toggle('active', btn.dataset.value === source);
            }
        });
        
        return source;
    }
    
    setAccelerator(type) {
        this.selectedAccelerator = type;
        
        // Update UI
        document.querySelectorAll('.segmented-control .segment').forEach(btn => {
            if (btn.closest('.control-section').querySelector('.control-label').textContent.includes('Accelerator')) {
                btn.classList.toggle('active', btn.dataset.value === type);
            }
        });
        
        return type;
    }
    
    async applyFilter() {
        if (this.currentState === 'streaming') return;
        
        const filters = this.filterManager.getSelectedFilters();
        const filterParam = filters.includes('grayscale') ? 'grayscale' : 'none';
        const grayscaleType = this.filterManager.getGrayscaleType();
        
        const startTime = performance.now();
        this.heroImage.classList.add('loading');
        
        try {
            const url = `/?filter=${filterParam}&accelerator=${this.selectedAccelerator}&grayscale_type=${grayscaleType}`;
            const response = await fetch(url);
            const html = await response.text();
            
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const newImage = doc.querySelector('#heroImage');
            
            if (newImage) {
                this.heroImage.src = newImage.src;
            }
            
            const processingTime = performance.now() - startTime;
            this.statsManager.statTime.textContent = processingTime.toFixed(0) + 'ms';
            
            const newUrl = new URL(window.location);
            newUrl.searchParams.set('filter', filterParam);
            newUrl.searchParams.set('accelerator', this.selectedAccelerator);
            newUrl.searchParams.set('grayscale_type', grayscaleType);
            window.history.pushState({}, '', newUrl);
            
        } catch (error) {
            console.error('Error applying filter:', error);
            this.toastManager.error('Filter Error', 'Failed to apply filter. Please try again.');
            this.statsManager.updateWebSocketStatus('error', 'Error processing');
        } finally {
            this.heroImage.classList.remove('loading');
        }
    }
}

