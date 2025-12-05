import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fixture, html } from '@open-wc/testing-helpers';
import './app-root';
import type { AppRoot } from './app-root';
import type {
  IConfigService,
  ILogger,
  IInputSourceService,
  IProcessorCapabilitiesService,
  IToolsService,
} from '../../application/di';

describe('AppRoot', () => {
  let element: AppRoot;
  let mockLogger: ILogger;
  let mockConfigService: IConfigService;
  let mockInputSourceService: IInputSourceService;
  let mockProcessorCapabilitiesService: IProcessorCapabilitiesService;
  let mockToolsService: IToolsService;

  beforeEach(async () => {
    mockLogger = {
      info: vi.fn(),
      debug: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
      initialize: vi.fn(),
      shutdown: vi.fn(),
    } as any;

    mockConfigService = {
      initialize: vi.fn(),
      getLogLevel: vi.fn().mockReturnValue('info'),
      getConsoleLogging: vi.fn().mockReturnValue(true),
    } as any;

    mockInputSourceService = {
      initialize: vi.fn(),
      getSources: vi.fn().mockReturnValue([]),
      getDefaultSource: vi.fn().mockReturnValue(null),
      isInitialized: vi.fn().mockReturnValue(true),
      listAvailableImages: vi.fn().mockResolvedValue([]),
    } as any;

    mockProcessorCapabilitiesService = {
      initialize: vi.fn(),
      isInitialized: vi.fn().mockReturnValue(true),
      getFilters: vi.fn().mockReturnValue([]),
      getFilterDefinitions: vi.fn().mockReturnValue([]),
      getGenericFilters: vi.fn().mockReturnValue([]),
      addFiltersUpdatedListener: vi.fn(),
      removeFiltersUpdatedListener: vi.fn(),
    } as any;

    mockToolsService = {
      initialize: vi.fn(),
      isInitialized: vi.fn().mockReturnValue(true),
      getCategories: vi.fn().mockReturnValue([]),
    } as any;

    element = await fixture<AppRoot>(html`<app-root></app-root>`);
    element.logger = mockLogger;
    element.configService = mockConfigService;
    element.inputSourceService = mockInputSourceService;
    element.processorCapabilitiesService = mockProcessorCapabilitiesService;
    element.toolsService = mockToolsService;
  });

  describe('Rendering', () => {
    it('should render control sections', async () => {
      await element.updateComplete;

      const controlSections = element.shadowRoot!.querySelectorAll('.control-section');
      expect(controlSections.length).toBeGreaterThan(0);
    });

    it('should render selected source indicator', async () => {
      await element.updateComplete;

      const badge = element.shadowRoot!.querySelector('.source-badge');
      expect(badge).toBeTruthy();
      expect(badge!.textContent).toBe('1');
    });

    it('should render accelerator controls', async () => {
      await element.updateComplete;

      const segments = element.shadowRoot!.querySelectorAll('.segment');
      expect(segments.length).toBe(2);
      expect(segments[0].textContent?.trim()).toBe('GPU');
      expect(segments[1].textContent?.trim()).toBe('CPU');
    });

    it('should render resolution select', async () => {
      await element.updateComplete;

      const select = element.shadowRoot!.querySelector('.compact-select');
      expect(select).toBeTruthy();
      expect(select!.querySelectorAll('option').length).toBe(3);
    });

    it('should mark GPU as active by default', async () => {
      await element.updateComplete;

      const gpuButton = element.shadowRoot!.querySelector('.segment[data-value="gpu"]');
      expect(gpuButton!.classList.contains('active')).toBe(true);
    });
  });

  describe('Initialization', () => {
    it('should throw error if logger is not provided', async () => {
      element.logger = undefined;

      await expect(element.initialize()).rejects.toThrow('Logger service not provided');
    });

    it('should log initialization messages', async () => {
      const mockToast = { configure: vi.fn() } as any;
      const mockStats = { reset: vi.fn() } as any;
      const mockFilter = { addEventListener: vi.fn() } as any;
      const mockVideoGrid = { setManagers: vi.fn(), addEventListener: vi.fn() } as any;
      const mockSourceDrawer = { addEventListener: vi.fn() } as any;
      const mockToolsDropdown = {} as any;
      const mockImageSelectorModal = { addEventListener: vi.fn() } as any;
      const mockTour = { startIfNeeded: vi.fn() } as any;
      const mockFab = { addEventListener: vi.fn() } as any;

      vi.spyOn(document, 'querySelector').mockImplementation((selector: string) => {
        if (selector === 'toast-container') return mockToast;
        if (selector === 'stats-panel') return mockStats;
        if (selector === 'filter-panel') return mockFilter;
        if (selector === 'video-grid') return mockVideoGrid;
        if (selector === 'source-drawer') return mockSourceDrawer;
        if (selector === 'tools-dropdown') return mockToolsDropdown;
        if (selector === 'image-selector-modal') return mockImageSelectorModal;
        if (selector === 'app-tour') return mockTour;
        if (selector === 'add-source-fab') return mockFab;
        return null;
      });

      await element.initialize();

      expect(mockLogger.info).toHaveBeenCalledWith('Initializing app-root...');
      expect(mockLogger.info).toHaveBeenCalledWith('App-root initialized', expect.any(Object));

      vi.restoreAllMocks();
    });

    it('should setup components during initialization', async () => {
      const mockToast = { configure: vi.fn() } as any;
      const mockStats = { reset: vi.fn() } as any;
      const mockFilter = { addEventListener: vi.fn() } as any;
      const mockVideoGrid = { setManagers: vi.fn(), addEventListener: vi.fn() } as any;
      const mockSourceDrawer = { addEventListener: vi.fn() } as any;
      const mockImageSelectorModal = { addEventListener: vi.fn() } as any;
      const mockFab = { addEventListener: vi.fn() } as any;

      vi.spyOn(document, 'querySelector').mockImplementation((selector: string) => {
        if (selector === 'toast-container') return mockToast;
        if (selector === 'stats-panel') return mockStats;
        if (selector === 'filter-panel') return mockFilter;
        if (selector === 'video-grid') return mockVideoGrid;
        if (selector === 'source-drawer') return mockSourceDrawer;
        if (selector === 'image-selector-modal') return mockImageSelectorModal;
        if (selector === 'add-source-fab') return mockFab;
        return null;
      });

      await element.initialize();

      expect(mockToast.configure).toHaveBeenCalledWith({ duration: 7000 });
      expect(mockVideoGrid.setManagers).toHaveBeenCalledWith(mockStats, mockToast);

      vi.restoreAllMocks();
    });

    it('should register filter update listener when service is set', async () => {
      await element.updateComplete;

      expect(mockProcessorCapabilitiesService.addFiltersUpdatedListener).toHaveBeenCalledTimes(1);
      const handler = vi.mocked(mockProcessorCapabilitiesService.addFiltersUpdatedListener).mock.calls[0][0];
      expect(typeof handler).toBe('function');
    });

    it('should remove filter update listener on disconnect', async () => {
      await element.updateComplete;

      element.disconnectedCallback();

      expect(mockProcessorCapabilitiesService.removeFiltersUpdatedListener).toHaveBeenCalledTimes(1);
    });
  });

  describe('Accelerator Control', () => {
    it('should update selected accelerator when GPU is clicked', async () => {
      await element.updateComplete;

      const cpuButton = element.shadowRoot!.querySelector('.segment[data-value="cpu"]') as HTMLButtonElement;
      cpuButton.click();
      await element.updateComplete;

      expect(cpuButton.classList.contains('active')).toBe(true);

      const gpuButton = element.shadowRoot!.querySelector('.segment[data-value="gpu"]') as HTMLButtonElement;
      expect(gpuButton.classList.contains('active')).toBe(false);
    });

    it('should update selected accelerator when CPU is clicked', async () => {
      await element.updateComplete;

      const cpuButton = element.shadowRoot!.querySelector('.segment[data-value="cpu"]') as HTMLButtonElement;
      cpuButton.click();
      await element.updateComplete;

      expect(cpuButton.classList.contains('active')).toBe(true);
    });
  });

  describe('Resolution Control', () => {
    it('should update selected resolution when changed', async () => {
      await element.updateComplete;

      const select = element.shadowRoot!.querySelector('.compact-select') as HTMLSelectElement;
      select.value = 'half';
      select.dispatchEvent(new Event('change'));
      await element.updateComplete;

      expect(select.value).toBe('half');
    });
  });

  describe('Source Selection', () => {
    it('should update source indicator with new values', async () => {
      await element.updateComplete;

      const badge = element.shadowRoot!.querySelector('.source-badge');
      expect(badge!.textContent).toBe('1');

      element['selectedSourceNumber'] = 2;
      element['selectedSourceName'] = 'Test Image';
      await element.updateComplete;

      expect(badge!.textContent).toBe('2');
    });
  });

  describe('Error Handling', () => {
    it('should handle missing services gracefully', async () => {
      element.inputSourceService = undefined;
      element.processorCapabilitiesService = undefined;
      element.toolsService = undefined;

      const mockToast = { configure: vi.fn() } as any;
      const mockStats = { reset: vi.fn() } as any;
      const mockFilter = { addEventListener: vi.fn() } as any;
      const mockVideoGrid = { setManagers: vi.fn(), addEventListener: vi.fn() } as any;
      const mockSourceDrawer = { addEventListener: vi.fn() } as any;
      const mockImageSelectorModal = { addEventListener: vi.fn() } as any;
      const mockFab = { addEventListener: vi.fn() } as any;

      vi.spyOn(document, 'querySelector').mockImplementation((selector: string) => {
        if (selector === 'toast-container') return mockToast;
        if (selector === 'stats-panel') return mockStats;
        if (selector === 'filter-panel') return mockFilter;
        if (selector === 'video-grid') return mockVideoGrid;
        if (selector === 'source-drawer') return mockSourceDrawer;
        if (selector === 'image-selector-modal') return mockImageSelectorModal;
        if (selector === 'add-source-fab') return mockFab;
        return null;
      });

      await expect(element.initialize()).resolves.not.toThrow();

      vi.restoreAllMocks();
    });
  });
});

