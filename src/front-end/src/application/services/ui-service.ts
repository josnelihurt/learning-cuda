import type { StatsPanel } from '../../components/app/stats-panel';
import type { CameraPreview } from '../../components/video/camera-preview';
import type { FilterPanel } from '../../components/app/filter-panel';
import type { ToastContainer } from '../../components/app/toast-container';
import type { WebSocketService } from '../../infrastructure/transport/websocket-frame-transport';
import { telemetryService } from '../../infrastructure/observability/telemetry-service';
import { logger } from '../../infrastructure/observability/otel-logger';
import type { IUIService } from '../../domain/interfaces/IUIService';
import type { ActiveFilterState } from '../../components/app/filter-panel.types';
import { FilterData } from '../../domain/value-objects';

declare const __APP_VERSION__: string;
declare const __APP_BRANCH__: string;
declare const __BUILD_TIME__: string;

export class UIService implements IUIService {
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
      this.infoBtn.addEventListener('mouseenter', () => {
        this.loadVersionInfo();
        if (this.infoTooltip) {
          this.infoTooltip.style.display = 'block';
        }
      });

      this.infoBtn.addEventListener('mouseleave', () => {
        if (this.infoTooltip) {
          this.infoTooltip.style.display = 'none';
        }
      });

      this.infoTooltip.addEventListener('mouseenter', () => {
        if (this.infoTooltip) {
          this.infoTooltip.style.display = 'block';
        }
      });

      this.infoTooltip.addEventListener('mouseleave', () => {
        if (this.infoTooltip) {
          this.infoTooltip.style.display = 'none';
        }
      });
    }
  }

  private async loadVersionInfo(): Promise<void> {
    try {
      const response = await fetch('/api/processor/capabilities');
      const data = await response.json();
      
      if (data.capabilities) {
        const cppVersionEl = document.getElementById('cppVersion');
        const goVersionEl = document.getElementById('goVersion');
        const jsVersionEl = document.getElementById('jsVersion');
        const branchVersionEl = document.getElementById('branchVersion');
        const buildVersionEl = document.getElementById('buildVersion');

        if (cppVersionEl) {
          cppVersionEl.textContent = data.capabilities.libraryVersion || '?';
        }
        if (goVersionEl) {
          goVersionEl.textContent = data.capabilities.apiVersion || '?';
        }
        if (jsVersionEl) {
          jsVersionEl.textContent = __APP_VERSION__;
        }
        if (branchVersionEl) {
          branchVersionEl.textContent = __APP_BRANCH__;
        }
        if (buildVersionEl) {
          buildVersionEl.textContent = new Date(__BUILD_TIME__).toLocaleString();
        }
      }
    } catch (e) {
      console.warn('Failed to load backend versions', e);
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
          full: { width: 640, height: 480 },
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

    document.querySelectorAll('.segmented-control .segment').forEach((btn) => {
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

    document.querySelectorAll('.segmented-control .segment').forEach((btn) => {
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

    const filtersState = this.normalizeFilters(this.filterManager.getActiveFilters());
    const filterParam = filtersState[0]?.id || 'none';
    const filterData = this.mapFiltersToValueObjects(filtersState);
    const grayscaleType = this.extractGrayscaleAlgorithm(filtersState);

    this.heroImage.classList.add('loading');

    await telemetryService.withSpanAsync(
      'UI.applyFilter',
      {
        filter: filterParam,
        accelerator: this.selectedAccelerator,
        grayscale_type: grayscaleType,
        input_source: this.selectedInputSource,
      },
      async (span) => {
        try {
          span?.addEvent('Creating canvas');
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          if (!ctx) {
            throw new Error('Canvas context not available');
          }

          const img = this.heroImage!;
          canvas.width = img.naturalWidth || 512;
          canvas.height = img.naturalHeight || 512;

          span?.setAttribute('image.width', canvas.width);
          span?.setAttribute('image.height', canvas.height);

          span?.addEvent('Drawing image to canvas');
          ctx.drawImage(img, 0, 0);

          span?.addEvent('Encoding image data');
          const imageData = canvas.toDataURL('image/png');

          span?.addEvent('Sending frame to backend');
          await this.wsService.sendSingleFrame(
            imageData,
            canvas.width,
            canvas.height,
            filterData,
            this.selectedAccelerator
          );

          span?.addEvent('Updating browser history');
          const newUrl = new URL(window.location.href);
          newUrl.searchParams.set('filter', filterParam);
          newUrl.searchParams.set('accelerator', this.selectedAccelerator);
          window.history.pushState({}, '', newUrl);
        } catch (error) {
          logger.error('Error applying filter', {
            'error.message': error instanceof Error ? error.message : String(error),
          });
          this.toastManager.error('Filter Error', 'Failed to apply filter. Please try again.');
          this.statsManager.updateWebSocketStatus('disconnected', 'Error processing');
          throw error;
        } finally {
          if (this.heroImage) {
            this.heroImage.classList.remove('loading');
          }
        }
      }
    );
  }

  private normalizeFilters(filters: ActiveFilterState[]): ActiveFilterState[] {
    if (!filters || filters.length === 0) {
      return [{ id: 'none', parameters: {} }];
    }
    return filters.map((filter) => ({
      id: filter.id,
      parameters: { ...filter.parameters },
    }));
  }

  private mapFiltersToValueObjects(filters: ActiveFilterState[]): FilterData[] {
    return filters.map((filter) => new FilterData(filter.id, { ...filter.parameters }));
  }

  private extractGrayscaleAlgorithm(filters: ActiveFilterState[]): string {
    const grayscale = filters.find((filter) => filter.id === 'grayscale');
    return (grayscale?.parameters.algorithm as string) || 'bt601';
  }
}
