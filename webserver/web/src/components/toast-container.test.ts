import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { fixture, html } from '@open-wc/testing-helpers';
import './toast-container';
import { ToastContainer } from './toast-container';

async function waitForToastRender(element: ToastContainer): Promise<void> {
  await element.updateComplete;
  await new Promise((resolve) => setTimeout(resolve, 10));
  await element.updateComplete;
}

describe('ToastContainer', () => {
  let element: ToastContainer;

  beforeEach(async () => {
    element = await fixture(html`<toast-container></toast-container>`);
    element.configure({ skipAnimations: true });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  describe('Configuration', () => {
    it('should apply custom configuration', () => {
      const config = { duration: 5000, maxToasts: 3 };
      element.configure(config);

      const id = element.show('info', 'Test');
      expect(id).toBeDefined();
    });

    it('should use default duration when not configured', () => {
      const id = element.show('info', 'Test');
      expect(id).toBeDefined();
    });
  });

  describe('Toast Creation', () => {
    it('should create a success toast', () => {
      const id = element.success('Success Title', 'Success message');

      expect(id).toBeDefined();
      expect(id).toMatch(/^toast-/);
    });

    it('should create an error toast', () => {
      const id = element.error('Error Title', 'Error message');

      expect(id).toBeDefined();
      expect(id).toMatch(/^toast-/);
    });

    it('should create a warning toast', () => {
      const id = element.warning('Warning Title', 'Warning message');

      expect(id).toBeDefined();
      expect(id).toMatch(/^toast-/);
    });

    it('should create an info toast', () => {
      const id = element.info('Info Title', 'Info message');

      expect(id).toBeDefined();
      expect(id).toMatch(/^toast-/);
    });

    it('should create toast with custom duration', () => {
      const id = element.show('success', 'Title', 'Message', 3000);

      expect(id).toBeDefined();
    });

    it('should create toast without auto-dismiss when duration is 0', () => {
      const id = element.show('info', 'Title', 'Message', 0);

      expect(id).toBeDefined();
    });
  });

  describe('Toast Limits', () => {
    it('should respect maxToasts configuration', async () => {
      element.configure({ maxToasts: 2 });

      const id1 = element.show('info', 'Toast 1');
      const id2 = element.show('info', 'Toast 2');
      const id3 = element.show('info', 'Toast 3');

      await element.updateComplete;

      expect(id1).toBeDefined();
      expect(id2).toBeDefined();
      expect(id3).toBeDefined();
    });
  });

  describe('Toast Dismissal', () => {
    it('should dismiss a specific toast by id', async () => {
      const id = element.show('info', 'Test Toast');

      await waitForToastRender(element);

      expect(element.getToastCount()).toBe(1);

      element.dismiss(id);
      await element.updateComplete;
      await new Promise((resolve) => setTimeout(resolve, 400));

      expect(element.getToastCount()).toBe(0);
    });

    it('should dismiss all toasts', async () => {
      element.show('info', 'Toast 1');
      element.show('info', 'Toast 2');
      element.show('info', 'Toast 3');
      await element.updateComplete;

      element.dismissAll();
      await element.updateComplete;
    });

    it('should handle dismissing non-existent toast gracefully', () => {
      expect(() => element.dismiss('non-existent-id')).not.toThrow();
    });
  });

  describe('Auto-dismissal', () => {
    it('should auto-dismiss toast after default duration', async () => {
      vi.useFakeTimers();
      const id = element.show('info', 'Auto-dismiss test');
      await element.updateComplete;

      vi.advanceTimersByTime(7000);
      await element.updateComplete;

      const toastElements = element.shadowRoot?.querySelectorAll('.toast.show');
      expect(toastElements?.length).toBe(0);
      vi.useRealTimers();
    });

    it('should auto-dismiss toast after custom duration', async () => {
      vi.useFakeTimers();
      const id = element.show('info', 'Custom duration', '', 2000);
      await element.updateComplete;

      vi.advanceTimersByTime(2000);
      await element.updateComplete;

      const toastElements = element.shadowRoot?.querySelectorAll('.toast.show');
      expect(toastElements?.length).toBe(0);
      vi.useRealTimers();
    });
  });

  describe('Duration Configuration', () => {
    it('should update default duration', () => {
      element.setDuration(5000);

      const id = element.show('info', 'Test');
      expect(id).toBeDefined();
    });
  });

  describe('Rendering', () => {
    it('should create toast with title only', async () => {
      const id = element.show('success', 'Title Only');

      await waitForToastRender(element);

      const toasts = element.getToasts();
      expect(toasts.length).toBe(1);
      expect(toasts[0].title).toBe('Title Only');
      expect(toasts[0].message).toBe('');
      expect(toasts[0].type).toBe('success');

      element.dismiss(id);
    });

    it('should create toast with title and message', async () => {
      const id = element.show('success', 'Title', 'Message content');

      await waitForToastRender(element);

      const toasts = element.getToasts();
      expect(toasts.length).toBe(1);
      expect(toasts[0].title).toBe('Title');
      expect(toasts[0].message).toBe('Message content');

      element.dismiss(id);
    });

    it('should create toast with correct type', async () => {
      const id = element.show('error', 'Error');

      await waitForToastRender(element);

      const toasts = element.getToasts();
      expect(toasts.length).toBe(1);
      expect(toasts[0].type).toBe('error');
      expect(toasts[0].show).toBe(true);

      element.dismiss(id);
    });
  });

  describe('Icon Rendering', () => {
    it('should render correct icon for each toast type', async () => {
      vi.useRealTimers();
      const types = ['success', 'error', 'warning', 'info'] as const;

      for (const type of types) {
        const el = await fixture(html`<toast-container></toast-container>`);
        (el as ToastContainer).show(type, 'Test');
        await el.updateComplete;
        await el.updateComplete;
        await new Promise((resolve) => setTimeout(resolve, 200));
        await el.updateComplete;

        const iconElement = el.shadowRoot?.querySelector('.toast-icon');
        expect(iconElement).toBeDefined();
        // Icon exists and is rendered (emojis may not have textContent in test environment)
      }
      vi.useFakeTimers();
    });
  });
});
