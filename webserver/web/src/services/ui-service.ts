import type { StatsPanel } from '../components/stats-panel';
import type { CameraPreview } from '../components/camera-preview';
import type { FilterPanel } from '../components/filter-panel';
import type { ToastContainer } from '../components/toast-container';
import type { WebSocketService } from './websocket-service';

export class UIService {
    selectedInputSource = 'lena';
    selectedAccelerator = 'gpu';
    currentState: 'static' | 'streaming' = 'static';

    private heroImage: HTMLImageElement | null = null;
    private resolutionSelect: HTMLSelectElement | null = null;
    private infoBtn: HTMLButtonElement | null = null;
    private infoTooltip: HTMLDivElement | null = null;

    constructor(
        private statsManager: StatsPanel,
        private cameraManager: CameraPreview,
        private filterManager: FilterPanel,
        private toastManager: ToastContainer,
        private wsService: WebSocketService
    ) {
        this.heroImage = document.getElementById('heroImage') as HTMLImageElement;
        this.resolutionSelect = document.getElementById('resolutionSelect') as HTMLSelectElement;
        this.infoBtn = document.getElementById('infoBtn') as HTMLButtonElement;
        this.infoTooltip = document.getElementById('infoTooltip') as HTMLDivElement;

        this.initInfoTooltip();
        this.initResolutionSelector();
    }

    private initInfoTooltip(): void {
        if (this.infoBtn && this.infoTooltip) {
            this.infoBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (this.infoTooltip) {
                    this.infoTooltip.style.display =
                        this.infoTooltip.style.display === 'none' ? 'block' : 'none';
                }
            });

            document.addEventListener('click', (e) => {
                if (this.infoBtn && this.infoTooltip &&
                    !this.infoBtn.contains(e.target as Node) &&
                    !this.infoTooltip.contains(e.target as Node)) {
                    this.infoTooltip.style.display = 'none';
                }
            });
        }
    }

    private initResolutionSelector(): void {
        if (this.resolutionSelect) {
            this.resolutionSelect.addEventListener('change', () => {
                if (!this.resolutionSelect) return;
                
                const resolution = this.resolutionSelect.value;
                const presets: Record<string, { width: number; height: number }> = {
                    quarter: { width: 160, height: 120 },
                    half: { width: 320, height: 240 },
                    full: { width: 640, height: 480 }
                };

                const preset = presets[resolution];
                if (preset) {
                    this.cameraManager.setResolution(preset.width, preset.height);
                }
            });
        }
    }

    setInputSource(source: string): string {
        this.selectedInputSource = source;

        document.querySelectorAll('.segmented-control .segment').forEach(btn => {
            const element = btn as HTMLButtonElement;
            const controlSection = btn.closest('.control-section');
            const label = controlSection?.querySelector('.control-label');
            
            if (label?.textContent?.includes('Input')) {
                element.classList.toggle('active', element.dataset.value === source);
            }
        });

        return source;
    }

    setAccelerator(type: string): string {
        this.selectedAccelerator = type;

        document.querySelectorAll('.segmented-control .segment').forEach(btn => {
            const element = btn as HTMLButtonElement;
            const controlSection = btn.closest('.control-section');
            const label = controlSection?.querySelector('.control-label');
            
            if (label?.textContent?.includes('Accelerator')) {
                element.classList.toggle('active', element.dataset.value === type);
            }
        });

        return type;
    }

    async applyFilter(): Promise<void> {
        if (this.currentState === 'streaming') return;
        if (!this.heroImage) return;

        const filters = this.filterManager.getSelectedFilters();
        const filterParam = filters.includes('grayscale') ? 'grayscale' : 'none';
        const grayscaleType = this.filterManager.getGrayscaleType();

        this.heroImage.classList.add('loading');

        try {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                throw new Error('Canvas context not available');
            }

            const img = this.heroImage;
            canvas.width = img.naturalWidth || 512;
            canvas.height = img.naturalHeight || 512;
            ctx.drawImage(img, 0, 0);

            const imageData = canvas.toDataURL('image/png');

            await this.wsService.sendSingleFrame(
                imageData,
                canvas.width,
                canvas.height,
                filters,
                this.selectedAccelerator,
                grayscaleType
            );

            const newUrl = new URL(window.location.href);
            newUrl.searchParams.set('filter', filterParam);
            newUrl.searchParams.set('accelerator', this.selectedAccelerator);
            newUrl.searchParams.set('grayscale_type', grayscaleType);
            window.history.pushState({}, '', newUrl);

        } catch (error) {
            console.error('Error applying filter:', error);
            this.toastManager.error('Filter Error', 'Failed to apply filter. Please try again.');
            this.statsManager.updateWebSocketStatus('disconnected', 'Error processing');
        } finally {
            if (this.heroImage) {
                this.heroImage.classList.remove('loading');
            }
        }
    }
}

