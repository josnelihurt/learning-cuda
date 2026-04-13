import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { fixture, html } from '@open-wc/testing-helpers';
import './filter-panel';
import { FilterPanel } from './filter-panel';
import type { Filter } from './filter-panel.types';
import './toast-container';

async function waitForRender(element: FilterPanel): Promise<void> {
  await element.updateComplete;
  await new Promise((resolve) => setTimeout(resolve, 10));
  await element.updateComplete;
}

function createMockFilter(overrides?: Partial<Filter>): Filter {
  return {
    id: 'test-filter',
    name: 'Test Filter',
    enabled: false,
    expanded: false,
    parameters: [],
    parameterValues: {},
    ...overrides,
  };
}

function createMockFilterWithNumberParam(overrides?: Partial<Filter>): Filter {
  return {
    id: 'sigma-filter',
    name: 'Gaussian Blur',
    enabled: false,
    expanded: false,
    parameters: [
      {
        id: 'sigma',
        name: 'Sigma',
        type: 'number',
        options: [],
        min: 0,
        max: 10,
        step: 0.1,
        defaultValue: '1.0',
      },
    ],
    parameterValues: { sigma: '1.0' },
    ...overrides,
  };
}

describe('FilterPanel', () => {
  let element: FilterPanel;
  let toastContainer: HTMLElement;

  beforeEach(async () => {
    vi.restoreAllMocks();
    
    if (!document.body.querySelector('toast-container')) {
      toastContainer = document.createElement('toast-container');
      document.body.appendChild(toastContainer);
    } else {
      toastContainer = document.body.querySelector('toast-container') as HTMLElement;
    }
    
    element = await fixture(html`<filter-panel></filter-panel>`);
  });

  afterEach(async () => {
    vi.restoreAllMocks();
  });

  describe('Card Toggle Behavior', () => {
    it('should expand and enable filter when opening a disabled filter', async () => {
      const filter = createMockFilter({ enabled: false, expanded: false });
      element.filters = [filter];

      await waitForRender(element);

      const header = element.shadowRoot?.querySelector('.filter-header') as HTMLElement;
      expect(header).toBeDefined();

      const dispatchSpy = vi.spyOn(element, 'dispatchEvent');
      header.click();

      await waitForRender(element);

      expect(element.filters[0].expanded).toBe(true);
      expect(element.filters[0].enabled).toBe(true);
      expect(dispatchSpy).toHaveBeenCalled();
    });

    it('should only expand filter when opening an already enabled filter', async () => {
      const filter = createMockFilter({ enabled: true, expanded: false });
      element.filters = [filter];

      await waitForRender(element);

      const header = element.shadowRoot?.querySelector('.filter-header') as HTMLElement;
      header.click();

      await waitForRender(element);

      expect(element.filters[0].expanded).toBe(true);
      expect(element.filters[0].enabled).toBe(true);
    });

    it('should collapse filter when closing an open filter', async () => {
      const filter = createMockFilter({ enabled: true, expanded: true });
      element.filters = [filter];

      await waitForRender(element);

      const header = element.shadowRoot?.querySelector('.filter-header') as HTMLElement;
      header.click();

      await waitForRender(element);

      expect(element.filters[0].expanded).toBe(false);
      expect(element.filters[0].enabled).toBe(true);
    });
  });

  describe('Checkbox Behavior', () => {
    it('should enable filter when checkbox is checked', async () => {
      const filter = createMockFilter({ enabled: false });
      element.filters = [filter];

      await waitForRender(element);

      const checkbox = element.shadowRoot?.querySelector(
        'input[type="checkbox"]'
      ) as HTMLInputElement;
      expect(checkbox).toBeDefined();

      checkbox.checked = true;
      checkbox.dispatchEvent(new Event('change', { bubbles: true }));

      await waitForRender(element);

      expect(element.filters[0].enabled).toBe(true);
    });

    it('should disable and collapse filter when checkbox is unchecked', async () => {
      const filter = createMockFilter({ enabled: true, expanded: true });
      element.filters = [filter];

      await waitForRender(element);

      const checkbox = element.shadowRoot?.querySelector(
        'input[type="checkbox"]'
      ) as HTMLInputElement;
      checkbox.checked = false;
      checkbox.dispatchEvent(new Event('change', { bubbles: true }));

      await waitForRender(element);

      expect(element.filters[0].enabled).toBe(false);
      expect(element.filters[0].expanded).toBe(false);
    });

    it('should not expand filter when enabling via checkbox', async () => {
      const filter = createMockFilter({ enabled: false, expanded: false });
      element.filters = [filter];

      await waitForRender(element);

      const checkbox = element.shadowRoot?.querySelector(
        'input[type="checkbox"]'
      ) as HTMLInputElement;
      checkbox.checked = true;
      checkbox.dispatchEvent(new Event('change', { bubbles: true }));

      await waitForRender(element);

      expect(element.filters[0].enabled).toBe(true);
      expect(element.filters[0].expanded).toBe(false);
    });
  });

  describe('Number Input Validation', () => {
    beforeEach(async () => {
      const filter = createMockFilterWithNumberParam({ enabled: true, expanded: true });
      element.filters = [filter];
      await waitForRender(element);
    });

    it('should clamp value to minimum when input is below min', async () => {
      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      expect(input).toBeDefined();

      input.value = '-5';
      input.dispatchEvent(new Event('input', { bubbles: true }));

      await waitForRender(element);

      expect(input.value).toBe('0');
      expect(element.filters[0].parameterValues.sigma).toBe('0');
    });

    it('should clamp value to maximum when input is above max', async () => {
      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      input.value = '15';
      input.dispatchEvent(new Event('input', { bubbles: true }));

      await waitForRender(element);

      expect(input.value).toBe('10');
      expect(element.filters[0].parameterValues.sigma).toBe('10');
    });

    it('should accept valid value within range', async () => {
      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      input.value = '5.5';
      input.dispatchEvent(new Event('input', { bubbles: true }));

      await waitForRender(element);

      expect(element.filters[0].parameterValues.sigma).toBe('5.5');
    });

    it('should prevent arrow up when value would exceed max', async () => {
      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      input.value = '10';
      input.focus();

      const keyDownEvent = new KeyboardEvent('keydown', {
        key: 'ArrowUp',
        bubbles: true,
        cancelable: true,
      });

      const preventDefaultSpy = vi.spyOn(keyDownEvent, 'preventDefault');
      input.dispatchEvent(keyDownEvent);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it('should prevent arrow down when value would go below min', async () => {
      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      input.value = '0';
      input.focus();

      const keyDownEvent = new KeyboardEvent('keydown', {
        key: 'ArrowDown',
        bubbles: true,
        cancelable: true,
      });

      const preventDefaultSpy = vi.spyOn(keyDownEvent, 'preventDefault');
      input.dispatchEvent(keyDownEvent);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it('should handle empty input gracefully', async () => {
      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      input.value = '';
      input.dispatchEvent(new Event('input', { bubbles: true }));

      await waitForRender(element);

      expect(element.filters[0].parameterValues.sigma).toBe('1.0');
    });

    it('should handle minus sign input gracefully', async () => {
      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      input.value = '-';
      input.dispatchEvent(new Event('input', { bubbles: true }));

      await waitForRender(element);

      expect(element.filters[0].parameterValues.sigma).toBe('1.0');
    });
  });

  describe('Number Input without Limits', () => {
    it('should accept any value when min and max are not defined', async () => {
      const filter = createMockFilter({
        id: 'no-limit-filter',
        name: 'No Limit Filter',
        enabled: true,
        expanded: true,
        parameters: [
          {
            id: 'value',
            name: 'Value',
            type: 'number',
            options: [],
          },
        ],
        parameterValues: { value: '5' },
      });
      element.filters = [filter];
      await waitForRender(element);

      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      input.value = '100';
      input.dispatchEvent(new Event('input', { bubbles: true }));

      await waitForRender(element);

      expect(element.filters[0].parameterValues.value).toBe('100');
    });
  });

  describe('Drag and Drop', () => {
    beforeEach(async () => {
      const filter1 = createMockFilter({ id: 'filter-1', name: 'Filter 1' });
      const filter2 = createMockFilter({ id: 'filter-2', name: 'Filter 2' });
      element.filters = [filter1, filter2];
      await waitForRender(element);
    });

    it('should only allow drag from drag-handle-container', async () => {
      const dragHandle = element.shadowRoot?.querySelector(
        '.drag-handle-container'
      ) as HTMLElement;
      expect(dragHandle).toBeDefined();
      expect(dragHandle.getAttribute('draggable')).toBe('true');

      const header = element.shadowRoot?.querySelector('.filter-header') as HTMLElement;
      expect(header.getAttribute('draggable')).toBeNull();
    });

    it('should prevent drag from checkbox container', async () => {
      const checkboxContainer = element.shadowRoot?.querySelector(
        '.checkbox-container'
      ) as HTMLElement;
      expect(checkboxContainer).toBeDefined();
      expect(checkboxContainer.getAttribute('draggable')).toBe('false');
    });

    it('should prevent drag from number inputs', async () => {
      const filter = createMockFilterWithNumberParam({ enabled: true, expanded: true });
      element.filters = [filter];
      await waitForRender(element);

      const input = element.shadowRoot?.querySelector(
        'input[type="number"]'
      ) as HTMLInputElement;
      expect(input.getAttribute('draggable')).toBe('false');
    });
  });

  describe('Filter Change Events', () => {
    it('should dispatch filter-change event when filter is toggled', async () => {
      const filter = createMockFilter({ enabled: false, expanded: false });
      element.filters = [filter];
      await waitForRender(element);

      const dispatchSpy = vi.spyOn(element, 'dispatchEvent');
      const header = element.shadowRoot?.querySelector('.filter-header') as HTMLElement;
      header.click();

      await waitForRender(element);

      expect(dispatchSpy).toHaveBeenCalled();
      const event = dispatchSpy.mock.calls[0][0] as CustomEvent;
      expect(event.type).toBe('filter-change');
    });

    it('should dispatch filter-change event when checkbox changes', async () => {
      const filter = createMockFilter({ enabled: false });
      element.filters = [filter];
      await waitForRender(element);

      const dispatchSpy = vi.spyOn(element, 'dispatchEvent');
      const checkbox = element.shadowRoot?.querySelector(
        'input[type="checkbox"]'
      ) as HTMLInputElement;
      checkbox.checked = true;
      checkbox.dispatchEvent(new Event('change', { bubbles: true }));

      await waitForRender(element);

      expect(dispatchSpy).toHaveBeenCalled();
      const event = dispatchSpy.mock.calls[0][0] as CustomEvent;
      expect(event.type).toBe('filter-change');
    });
  });

  describe('getActiveFilters', () => {
    it('should return only enabled filters', () => {
      const filter1 = createMockFilter({ id: 'filter-1', enabled: true });
      const filter2 = createMockFilter({ id: 'filter-2', enabled: false });
      element.filters = [filter1, filter2];

      const active = element.getActiveFilters();

      expect(active.length).toBe(1);
      expect(active[0].id).toBe('filter-1');
    });

    it('should return none filter when no filters are enabled', () => {
      const filter1 = createMockFilter({ id: 'filter-1', enabled: false });
      const filter2 = createMockFilter({ id: 'filter-2', enabled: false });
      element.filters = [filter1, filter2];

      const active = element.getActiveFilters();

      expect(active.length).toBe(1);
      expect(active[0].id).toBe('none');
    });
  });

  describe('setFilters', () => {
    it('should update filter states from active filter states', () => {
      const filter = createMockFilter({
        id: 'test-filter',
        enabled: false,
        expanded: false,
        parameterValues: {},
      });
      element.filters = [filter];

      const activeStates = [
        {
          id: 'test-filter',
          parameters: { param1: 'value1' },
        },
      ];

      element.setFilters(activeStates);

      expect(element.filters[0].enabled).toBe(true);
      expect(element.filters[0].expanded).toBe(true);
      expect(element.filters[0].parameterValues.param1).toBe('value1');
    });
  });
});
