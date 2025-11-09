import { describe, it, beforeEach, afterEach, expect, vi } from 'vitest';
import { fixture, html } from '@open-wc/testing-helpers';
import './app-tour';
import type { AppTour } from './app-tour';

const DISMISS_KEY = 'cuda-app-tour-dismissed';

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

    const nextButton = tour.shadowRoot?.querySelector('.primary') as HTMLButtonElement;
    nextButton.click();
    await tour.updateComplete;

    const secondTitle = tour.shadowRoot?.querySelector('.step-title')?.textContent?.trim();
    expect(secondTitle).toContain('Filter Panel');

    nextButton.click();
    await tour.updateComplete;

    const thirdTitle = tour.shadowRoot?.querySelector('.step-title')?.textContent?.trim();
    expect(thirdTitle).toContain('Switch Images');

    nextButton.click();
    await tour.updateComplete;

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

    tour.startIfNeeded();
    await tour.updateComplete;

    const overlay = tour.shadowRoot?.querySelector('.overlay');
    expect(overlay).toBeNull();
  });
});

