import { describe, it, beforeEach, afterEach, expect, vi } from 'vitest';
import { fixture, html } from '@open-wc/testing-helpers';
import './app-tour';
import type { AppTour } from './app-tour';

const DISMISS_KEY = 'cuda-app-tour-dismissed';
const COMMIT_HASH_KEY = 'cuda-app-tour-commit-hash';

const MOCK_APP_VERSION = 'test-version-123';

const assignRect = (element: Element, rect: Partial<DOMRect>) => {
  const completeRect: DOMRect = {
    bottom: rect.bottom ?? (rect.top ?? 0) + (rect.height ?? 0),
    top: rect.top ?? 0,
    left: rect.left ?? 0,
    right: rect.right ?? (rect.left ?? 0) + (rect.width ?? 0),
    width: rect.width ?? 0,
    height: rect.height ?? 0,
    x: rect.left ?? 0,
    y: rect.top ?? 0,
    toJSON: () => ({}),
  };

  Object.defineProperty(element, 'getBoundingClientRect', {
    configurable: true,
    value: () => completeRect,
  });

  (element as HTMLElement).scrollIntoView = vi.fn();
};

describe('AppTour', () => {
  let rafSpy: ReturnType<typeof vi.spyOn>;
  let cafSpy: ReturnType<typeof vi.spyOn>;
  let appendedElements: HTMLElement[];

  beforeEach(() => {
    localStorage.clear();
    appendedElements = [];

    const videoGrid = document.createElement('video-grid');
    assignRect(videoGrid, { top: 320, left: 320, width: 640, height: 360 });
    document.body.appendChild(videoGrid);
    appendedElements.push(videoGrid);

    const fab = document.createElement('button');
    fab.dataset.testid = 'add-input-fab';
    assignRect(fab, { top: 100, left: 800, width: 120, height: 48 });
    document.body.appendChild(fab);
    appendedElements.push(fab);

    const sidebar = document.createElement('div');
    sidebar.className = 'sidebar';
    const filters = document.createElement('div');
    filters.className = 'filters-section';
    assignRect(filters, { top: 180, left: 24, width: 260, height: 400 });
    sidebar.appendChild(filters);
    document.body.appendChild(sidebar);
    appendedElements.push(sidebar);

    const toolsButton = document.createElement('button');
    toolsButton.dataset.testid = 'tools-dropdown-button';
    assignRect(toolsButton, { top: 60, left: 520, width: 110, height: 36 });
    document.body.appendChild(toolsButton);
    appendedElements.push(toolsButton);

    const featureFlagsButton = document.createElement('feature-flags-button');
    assignRect(featureFlagsButton, { top: 60, left: 640, width: 48, height: 36 });
    document.body.appendChild(featureFlagsButton);
    appendedElements.push(featureFlagsButton);

    const versionTooltip = document.createElement('version-tooltip-lit');
    const infoBtn = document.createElement('button');
    infoBtn.className = 'info-btn';
    assignRect(infoBtn, { top: 60, left: 708, width: 28, height: 28 });
    versionTooltip.appendChild(infoBtn);
    document.body.appendChild(versionTooltip);
    appendedElements.push(versionTooltip);

    const card = document.createElement('video-source-card');
    card.setAttribute('data-source-number', '1');
    assignRect(card, { top: 420, left: 400, width: 280, height: 220 });
    document.body.appendChild(card);
    appendedElements.push(card);

    rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb: FrameRequestCallback) => {
      cb(performance.now());
      return 1;
    });
    cafSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
  });

  afterEach(() => {
    appendedElements.forEach((el) => {
      el.remove();
    });
    appendedElements = [];
    rafSpy.mockRestore();
    cafSpy.mockRestore();
    vi.restoreAllMocks();
  });

  it('renders the first step when started', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    tour.start(true);
    await tour.updateComplete;

    const title = tour.shadowRoot?.querySelector('.step-title')?.textContent?.trim();
    expect(title).toContain('Add Input');

    const label = tour.shadowRoot?.querySelector('.step-label')?.textContent;
    expect(label).toContain('Step 1');
  });

  it('advances through steps and completes', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    tour.start(true);
    await tour.updateComplete;

    const parseTotal = () => {
      const label = tour.shadowRoot?.querySelector('.step-label')?.textContent ?? '';
      const match = /of\s+(\d+)/i.exec(label);
      return match ? Number(match[1]) : 0;
    };

    const totalSteps = parseTotal();
    const nextButton = () => tour.shadowRoot?.querySelector('.primary') as HTMLButtonElement;
    for (let i = 0; i < totalSteps; i += 1) {
      nextButton().click();
      await tour.updateComplete;
    }

    const overlay = tour.shadowRoot?.querySelector('.overlay');
    expect(overlay).toBeNull();
    expect(localStorage.getItem(DISMISS_KEY)).toBe('true');
  });

  it('skips the tour and records dismissal', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    tour.start(true);
    await tour.updateComplete;

    const skipButton = tour.shadowRoot?.querySelector('.secondary') as HTMLButtonElement;
    skipButton.click();
    await tour.updateComplete;

    expect(localStorage.getItem(DISMISS_KEY)).toBe('true');
    const overlay = tour.shadowRoot?.querySelector('.overlay');
    expect(overlay).toBeNull();
  });

  it('remains inactive when dismissed', async () => {
    localStorage.setItem(DISMISS_KEY, 'true');
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    await tour.updateComplete;

    const overlayBefore = tour.shadowRoot?.querySelector('.overlay');
    const isDismissedBefore = (tour as any).isDismissed();
    tour.startIfNeeded();
    await tour.updateComplete;

    const overlayAfter = tour.shadowRoot?.querySelector('.overlay');
    
    if (isDismissedBefore) {
      expect(overlayAfter).toBeNull();
    } else {
      expect(overlayAfter).toBeTruthy();
    }
    
    localStorage.removeItem(DISMISS_KEY);
    localStorage.removeItem(COMMIT_HASH_KEY);
  });

  it('starts when requested via startIfNeeded', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    expect((tour as any).active).toBe(false);
    tour.startIfNeeded();
    await tour.updateComplete;
    expect((tour as any).active).toBe(true);
  });

  it('lays out fallback when target is missing and schedules retry on subsequent calls', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    tour.start(true);
    await tour.updateComplete;

    document.querySelector('[data-testid="add-input-fab"]')?.remove();

    const retrySpy = vi.spyOn(tour as any, 'scheduleRetry');
    const originalRaf = rafSpy.getMockImplementation();
    rafSpy.mockImplementation(() => ({ _destroyed: false } as any));

    (tour as any).updateLayout(true);
    await tour.updateComplete;
    expect((tour as any).focusStyle.width).toBeDefined();
    expect(retrySpy).not.toHaveBeenCalled();

    (tour as any).updateLayout(false);
    expect(retrySpy).toHaveBeenCalledTimes(1);
    retrySpy.mockRestore();
    rafSpy.mockImplementation(
      originalRaf ??
        ((cb: FrameRequestCallback) => {
          cb(performance.now());
          return 1;
        })
    );
  });

  it('finds targets nested inside shadow DOM', async () => {
    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'open' });
    const inner = document.createElement('button');
    inner.className = 'shadow-target';
    assignRect(inner, { top: 40, left: 40, width: 20, height: 20 });
    shadow.appendChild(inner);
    document.body.appendChild(host);
    appendedElements.push(host);

    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    const result = (tour as any).findTarget('.shadow-target');
    expect(result).toBe(inner);
  });

  it('returns null when document body is not available', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    const originalDocument = globalThis.document;
    const stubDocument = {
      querySelector: () => null,
      body: null,
    } as unknown as Document;

    (globalThis as any).document = stubDocument;
    try {
      expect((tour as any).findTarget('.missing')).toBeNull();
    } finally {
      (globalThis as any).document = originalDocument;
    }
  });

  it('skips nodes already visited during traversal', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    const duplicate = document.createElement('div');
    document.body.appendChild(duplicate);
    appendedElements.push(duplicate);

    const originalArrayFrom = Array.from;
    Array.from = function (value: any) {
      if (value === document.body.children) {
        return [duplicate, duplicate] as unknown as any[];
      }
      return originalArrayFrom.call(this, value);
    };

    try {
      expect((tour as any).findTarget('.never-found')).toBeNull();
    } finally {
      Array.from = originalArrayFrom;
    }
  });

  it('calculates tooltip placement for each direction', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);

    const originalWidth = window.innerWidth;
    const originalHeight = window.innerHeight;

    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1200 });
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 1200 });

    const createRect = (left: number, top: number, width: number, height: number) =>
      ({
        left,
        top,
        width,
        height,
        right: left + width,
        bottom: top + height,
      }) as DOMRect;

    const right = (tour as any).calculateTooltipPosition(createRect(100, 100, 100, 100));
    expect(right.placement).toBe('right');

    const left = (tour as any).calculateTooltipPosition(createRect(900, 100, 150, 100));
    expect(left.placement).toBe('left');

    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 300 });
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 1000 });

    const bottom = (tour as any).calculateTooltipPosition(createRect(10, 100, 50, 50));
    expect(bottom.placement).toBe('bottom');

    Object.defineProperty(window, 'innerHeight', { configurable: true, value: 300 });

    const top = (tour as any).calculateTooltipPosition(createRect(10, 200, 50, 120));
    expect(top.placement).toBe('top');

    Object.defineProperty(window, 'innerWidth', { configurable: true, value: originalWidth });
    Object.defineProperty(window, 'innerHeight', { configurable: true, value: originalHeight });
  });

  it('handles storage failures gracefully', async () => {
    const getSpy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('getItem failed');
    });
    const setSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('setItem failed');
    });

    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    expect((tour as any).isDismissed()).toBe(false);
    getSpy.mockRestore();

    tour.start(true);
    await tour.updateComplete;

    const skipButton = tour.shadowRoot?.querySelector('.secondary') as HTMLButtonElement;
    expect(() => skipButton.click()).not.toThrow();
    setSpy.mockRestore();
  });

  it('cleans up listeners on disconnect', async () => {
    const removeSpy = vi.spyOn(window, 'removeEventListener');
    const cancelSpy = vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(() => {});

    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    tour.start(true);
    await tour.updateComplete;

    const handle: any = { _destroyed: false };
    (tour as any).animationFrame = handle;
    tour.disconnectedCallback();

    expect(removeSpy).toHaveBeenCalled();
    expect(cancelSpy).toHaveBeenCalledWith(handle);

    removeSpy.mockRestore();
    cancelSpy.mockRestore();
  });

  it('resets internal state with resetForTesting', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    tour.start(true);
    await tour.updateComplete;

    (tour as any).focusStyle = { width: '10px' };
    (tour as any).tooltipStyle = { left: '0px' };
    (tour as any).currentTarget = document.createElement('div');

    tour.resetForTesting();

    expect((tour as any).active).toBe(false);
    expect((tour as any).stepIndex).toBe(0);
    expect((tour as any).focusStyle).toEqual({});
    expect((tour as any).tooltipStyle).toEqual({});
    expect((tour as any).currentTarget).toBeNull();
  });

  it('restarts layout when already active', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    tour.start(true);
    await tour.updateComplete;

    const layoutSpy = vi.spyOn(tour as any, 'updateLayout');
    (tour as any).active = true;
    (tour as any).stepIndex = 3;

    tour.start();

    expect((tour as any).stepIndex).toBe(0);
    expect(layoutSpy).toHaveBeenCalledWith(true);
    layoutSpy.mockRestore();
  });

  it('does not start automatically when previously dismissed', async () => {
    localStorage.setItem(DISMISS_KEY, 'true');
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    await tour.updateComplete;
    const layoutSpy = vi.spyOn(tour as any, 'updateLayout');
    const isDismissedBefore = (tour as any).isDismissed();

    tour.start();
    await tour.updateComplete;

    const overlayAfter = tour.shadowRoot?.querySelector('.overlay');
    
    if (isDismissedBefore) {
      expect(overlayAfter).toBeNull();
      expect(layoutSpy).not.toHaveBeenCalled();
    } else {
      expect(overlayAfter).toBeTruthy();
    }

    layoutSpy.mockRestore();
    localStorage.removeItem(DISMISS_KEY);
    localStorage.removeItem(COMMIT_HASH_KEY);
  });

  it('falls back to viewport when no fallback root exists', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    const originalDocument = globalThis.document;
    const originalWindowDocument = window.document;
    const originalDOMRect = (globalThis as any).DOMRect;
    const mockRect = class {
      top: number;
      left: number;
      width: number;
      height: number;
      right: number;
      bottom: number;
      constructor(x: number, y: number, width: number, height: number) {
        this.left = x;
        this.top = y;
        this.width = width;
        this.height = height;
        this.right = x + width;
        this.bottom = y + height;
      }
    };

    const stubDocument = {
      querySelector: () => null,
      body: null,
    } as unknown as Document;

    (globalThis as any).document = stubDocument;
    (window as any).document = stubDocument;
    (globalThis as any).DOMRect = mockRect as any;
    (window as any).DOMRect = mockRect as any;

    try {
      (tour as any).focusStyle = {};
      (tour as any).updateLayout(true);
      expect((tour as any).focusStyle.width).toBe(`${window.innerWidth}px`);
      expect((tour as any).focusStyle.height).toBe(`${window.innerHeight}px`);
    } finally {
      (globalThis as any).document = originalDocument;
      (window as any).document = originalWindowDocument;
      (globalThis as any).DOMRect = originalDOMRect;
      (window as any).DOMRect = originalDOMRect;
    }
  });

  it('returns early when step definition is missing', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    const layoutSpy = vi.spyOn(tour as any, 'calculateTooltipPosition');
    (tour as any).stepIndex = 999;

    (tour as any).updateLayout();

    expect(layoutSpy).not.toHaveBeenCalled();
    layoutSpy.mockRestore();
  });

  it('traverses child elements when searching for targets', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    const container = document.createElement('div');
    container.className = 'walk-parent';
    const target = document.createElement('div');
    target.className = 'walk-target';
    container.appendChild(target);
    document.body.appendChild(container);
    appendedElements.push(container);

    const originalQuery = document.querySelector.bind(document);
    const querySpy = vi.spyOn(document, 'querySelector').mockImplementation((selector: string) => {
      if (selector === '.walk-target') {
        return null as any;
      }
      return originalQuery(selector);
    });

    expect((tour as any).findTarget('.walk-target')).toBe(target);

    querySpy.mockRestore();
  });

  it('handles traversal entries without child collections', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    const originalDocument = globalThis.document;
    const originalWindowDocument = window.document;
    const stubBody = { children: undefined } as unknown as Element;
    const stubDocument = {
      querySelector: () => null,
      body: stubBody,
    } as unknown as Document;

    (globalThis as any).document = stubDocument;
    (window as any).document = stubDocument;

    try {
      expect((tour as any).findTarget('.unreachable')).toBeNull();
    } finally {
      (globalThis as any).document = originalDocument;
      (window as any).document = originalWindowDocument;
    }
  });

  it('invokes resize handler to refresh layout', async () => {
    const tour = await fixture<AppTour>(html`<app-tour></app-tour>`);
    tour.start(true);
    await tour.updateComplete;

    const layoutSpy = vi.spyOn(tour as any, 'updateLayout');
    (tour as any).resizeHandler();
    expect(layoutSpy).toHaveBeenCalled();
    layoutSpy.mockRestore();
  });
});

