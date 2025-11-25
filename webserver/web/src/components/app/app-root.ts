import { LitElement, html, css } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import type { PropertyValues } from 'lit';
import type {
  IConfigService,
  ITelemetryService,
  ILogger,
  IInputSourceService,
  IProcessorCapabilitiesService,
  IToolsService,
  IVideoService,
  IWebRTCService,
} from '../../application/di';
import type { VideoGrid } from '../video/video-grid';
import type { FilterPanel } from './filter-panel';
import type { SourceDrawer } from './source-drawer';
import type { ToolsDropdown } from '../ui/tools-dropdown';
import type { ImageSelectorModal } from '../image/image-selector-modal';
import type { AppTour } from './app-tour';
import type { ToastContainer } from './toast-container';
import type { StatsPanel } from './stats-panel';
import type { ActiveFilterState } from './filter-panel.types';

const TOUR_DISMISS_KEY = 'cuda-app-tour-dismissed';

@customElement('app-root')
export class AppRoot extends LitElement {
  @property({ type: Object }) configService?: IConfigService;
  @property({ type: Object }) telemetryService?: ITelemetryService;
  @property({ type: Object }) logger?: ILogger;
  @property({ type: Object }) inputSourceService?: IInputSourceService;
  @property({ type: Object }) processorCapabilitiesService?: IProcessorCapabilitiesService;
  @property({ type: Object }) toolsService?: IToolsService;
  @property({ type: Object }) videoService?: IVideoService;
  @property({ type: Object }) webrtcService?: IWebRTCService;

  @state() private selectedAccelerator = 'gpu';
  @state() private selectedResolution = 'original';
  @state() private selectedSourceNumber = 1;
  @state() private selectedSourceName = 'Lena';
  @state() private currentSourceNumberForImageChange: number | null = null;

  private toastManager: ToastContainer | null = null;
  private statsManager: StatsPanel | null = null;
  private filterManager: FilterPanel | null = null;
  private videoGrid: VideoGrid | null = null;
  private sourceDrawer: SourceDrawer | null = null;
  private toolsDropdown: ToolsDropdown | null = null;
  private imageSelectorModal: ImageSelectorModal | null = null;
  private tour: AppTour | null = null;
  private filtersUpdatedHandler = () => {
    if (this.filterManager && this.processorCapabilitiesService) {
      this.filterManager.filters = this.processorCapabilitiesService.getFilters();
    }
  };

  static styles = css`
    :host {
      display: block;
      width: 100%;
    }

    .sidebar-controls {
      display: block;
    }

    .control-section {
      padding: 16px 0;
      border-bottom: 1px solid var(--border-color);
    }

    .control-label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .selected-source-compact {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: var(--background-secondary);
      border-radius: 6px;
    }

    .source-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 24px;
      background: var(--primary-color);
      color: white;
      border-radius: 4px;
      font-weight: 600;
      font-size: 13px;
    }

    .segmented-control {
      display: flex;
      gap: 4px;
      background: var(--background-secondary);
      padding: 4px;
      border-radius: 8px;
    }

    .segment {
      flex: 1;
      padding: 8px 16px;
      background: transparent;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 500;
      color: var(--text-secondary);
      transition: all 0.2s;
    }

    .segment:hover {
      color: var(--text-primary);
    }

    .segment.active {
      background: white;
      color: var(--primary-color);
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .compact-select {
      width: 100%;
      padding: 10px 12px;
      border: 2px solid var(--border-color);
      border-radius: 6px;
      background: white;
      font-size: 14px;
      color: var(--text-primary);
      cursor: pointer;
      transition: border-color 0.2s;
    }

    .compact-select:hover {
      border-color: var(--primary-color);
    }

    .compact-select:focus {
      outline: none;
      border-color: var(--primary-color);
    }
  `;

  render() {
    return html`
      <div class="sidebar-controls">
        <div class="control-section">
          <label class="control-label">Selected</label>
          <div class="selected-source-compact">
            <span class="source-badge">${this.selectedSourceNumber}</span>
            <span>${this.selectedSourceName}</span>
          </div>
        </div>

        <div class="control-section">
          <label class="control-label">Accelerator</label>
          <div class="segmented-control">
            <button
              class="segment ${this.selectedAccelerator === 'gpu' ? 'active' : ''}"
              @click=${() => this.setAccelerator('gpu')}
              data-value="gpu"
            >
              GPU
            </button>
            <button
              class="segment ${this.selectedAccelerator === 'cpu' ? 'active' : ''}"
              @click=${() => this.setAccelerator('cpu')}
              data-value="cpu"
            >
              CPU
            </button>
          </div>
        </div>

        <div class="control-section">
          <label class="control-label">Resolution</label>
          <select
            class="compact-select"
            .value=${this.selectedResolution}
            @change=${this.handleResolutionChange}
            data-testid="resolution-select"
          >
            <option value="original">Original Size</option>
            <option value="half">Half</option>
            <option value="quarter">Quarter</option>
          </select>
        </div>
      </div>
    `;
  }

  async initialize(): Promise<void> {
    if (!this.logger) {
      throw new Error('Logger service not provided');
    }

    this.logger.info('Initializing app-root...');

    await this.setupComponents();
    this.setupEventListeners();
    this.loadDefaultSource();

    if (this.statsManager) {
      this.statsManager.reset();
    }

    this.logger.info('App-root initialized');

    await this.updateComplete;
    await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));

    if (this.tour) {
      this.tour.startIfNeeded();
    }
  }

  protected updated(changedProperties: PropertyValues<AppRoot>) {
    super.updated(changedProperties);
    if (changedProperties.has('processorCapabilitiesService')) {
      const previousService = changedProperties.get('processorCapabilitiesService') as IProcessorCapabilitiesService | undefined;
      previousService?.removeFiltersUpdatedListener(this.filtersUpdatedHandler);
      this.processorCapabilitiesService?.addFiltersUpdatedListener(this.filtersUpdatedHandler);
    }
  }

  disconnectedCallback(): void {
    this.processorCapabilitiesService?.removeFiltersUpdatedListener(this.filtersUpdatedHandler);
    super.disconnectedCallback();
  }

  private async setupComponents(): Promise<void> {
    this.toastManager = document.querySelector('toast-container');
    this.statsManager = document.querySelector('stats-panel');
    this.filterManager = document.querySelector('filter-panel');
    this.videoGrid = document.querySelector('video-grid');
    this.sourceDrawer = document.querySelector('source-drawer');
    this.toolsDropdown = document.querySelector('tools-dropdown');
    this.imageSelectorModal = document.querySelector('image-selector-modal');
    this.tour = document.querySelector('app-tour');

    this.handleAutomationTourDismissal();

    if (this.toastManager) {
      this.toastManager.configure({ duration: 7000 });
    }

    this.updateFiltersFromService();
    this.updateToolsFromService();

    if (this.videoGrid && this.statsManager && this.toastManager) {
      this.videoGrid.setManagers(this.statsManager, this.toastManager);
    }
  }

  private updateFiltersFromService(): void {
    if (this.filterManager && this.processorCapabilitiesService?.isInitialized()) {
      this.filterManager.filters = this.processorCapabilitiesService.getFilters();
    } else if (this.processorCapabilitiesService && !this.processorCapabilitiesService.isInitialized()) {
      this.processorCapabilitiesService.initialize().then(() => {
        if (this.filterManager) {
          this.filterManager.filters = this.processorCapabilitiesService!.getFilters();
        }
      }).catch((error) => {
        this.logger?.error('Failed to initialize processor capabilities', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      });
    }
  }

  private updateToolsFromService(): void {
    if (this.toolsDropdown && this.toolsService?.isInitialized()) {
      this.toolsDropdown.categories = this.toolsService.getCategories();
    }
  }

  private setupEventListeners(): void {
    if (this.filterManager) {
      this.filterManager.addEventListener(
        'filter-change',
        ((e: CustomEvent) => {
          if (this.videoGrid) {
            const filters: ActiveFilterState[] =
              e.detail?.filters ?? this.filterManager!.getActiveFilters();
            this.videoGrid.applyFilterToSelected(
              filters,
              this.selectedAccelerator,
              this.selectedResolution
            );
          }
        }) as EventListener
      );
    }

    if (this.videoGrid) {
      this.videoGrid.addEventListener('source-selection-changed', ((e: CustomEvent) => {
        this.updateSelectedSourceIndicator(e.detail.sourceNumber, e.detail.sourceId);

        if (this.filterManager && e.detail.filters !== undefined) {
          this.filterManager.setFilters(e.detail.filters);
        }

        if (e.detail.resolution) {
          this.selectedResolution = e.detail.resolution;
        }
      }) as EventListener);

      this.videoGrid.addEventListener('change-image-requested', (event) => {
        void this.handleChangeImageRequested(event as CustomEvent);
      });
    }

    const fab = document.querySelector('add-source-fab');
    if (fab) {
      fab.addEventListener('open-drawer', () => {
        if (this.sourceDrawer && this.videoGrid && this.inputSourceService) {
          const sources = this.inputSourceService.getSources();
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

    if (this.imageSelectorModal) {
      this.imageSelectorModal.addEventListener('image-selected', ((e: CustomEvent) => {
        if (this.videoGrid && this.currentSourceNumberForImageChange !== null) {
          const imagePath = e.detail.image.path;
          this.videoGrid.changeSourceImage(this.currentSourceNumberForImageChange, imagePath);
          this.currentSourceNumberForImageChange = null;
        }
      }) as EventListener);
    }
  }

  private loadDefaultSource(): void {
    if (!this.inputSourceService || !this.videoGrid) {
      return;
    }

    const defaultSource = this.inputSourceService.getDefaultSource();
    if (defaultSource) {
      this.videoGrid.addSource(defaultSource);
    }
  }

  private updateSelectedSourceIndicator(sourceNumber: number, sourceId: string): void {
    this.selectedSourceNumber = sourceNumber;

    if (this.videoGrid) {
      const source = this.videoGrid.getSources().find((s) => s.id === sourceId);
      if (source) {
        this.selectedSourceName = source.name;
      }
    }
  }

  private setAccelerator(type: string): void {
    this.selectedAccelerator = type;

    if (this.videoGrid && this.filterManager) {
      const filters = this.filterManager.getActiveFilters();
      this.videoGrid.applyFilterToSelected(filters, type, this.selectedResolution);
    }
  }

  private handleAutomationTourDismissal(): void {
    if (typeof window === 'undefined' || !this.tour) {
      return;
    }

    const globalScope = window as typeof window & { __ENABLE_TOUR__?: boolean };
    const isAutomation =
      typeof navigator !== 'undefined' &&
      Boolean((navigator as Navigator & { webdriver?: boolean }).webdriver);

    if (!isAutomation || globalScope.__ENABLE_TOUR__) {
      return;
    }

    try {
      window.localStorage.setItem(TOUR_DISMISS_KEY, 'true');
    } catch {
      // ignore storage errors (e.g. private mode, disabled storage)
    }
  }

  private handleResolutionChange(e: Event): void {
    const select = e.target as HTMLSelectElement;
    this.selectedResolution = select.value;

    if (this.videoGrid && this.filterManager) {
      const filters = this.filterManager.getActiveFilters();
      this.videoGrid.applyFilterToSelected(filters, this.selectedAccelerator, this.selectedResolution);
    }
  }

  private async handleChangeImageRequested(event: CustomEvent): Promise<void> {
    this.currentSourceNumberForImageChange = event.detail?.sourceNumber ?? null;
    if (this.imageSelectorModal && this.inputSourceService && this.currentSourceNumberForImageChange !== null) {
      const images = await this.inputSourceService.listAvailableImages();
      this.imageSelectorModal.open(images);
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    'app-root': AppRoot;
  }
}

