import { describe, it, expect, beforeEach } from 'vitest';
import { fixture, html } from '@open-wc/testing-helpers';
import './image-selector-modal';
import type { ImageSelectorModal } from './image-selector-modal';
import { StaticImage } from '../gen/config_service_pb';

describe('ImageSelectorModal', () => {
  let element: ImageSelectorModal;

  beforeEach(async () => {
    element = await fixture<ImageSelectorModal>(
      html`<image-selector-modal></image-selector-modal>`
    );
  });

  describe('Initialization', () => {
    it('should create element', () => {
      expect(element).toBeDefined();
      expect(element.tagName).toBe('IMAGE-SELECTOR-MODAL');
    });

    it('should be hidden by default', () => {
      expect(element.hasAttribute('open')).toBe(false);
    });
  });

  describe('Opening modal', () => {
    it('should show modal when open is called', async () => {
      const images = [
        new StaticImage({ id: 'lena', displayName: 'Lena', path: '/data/lena.png', isDefault: true }),
        new StaticImage({ id: 'mandrill', displayName: 'Mandrill', path: '/data/mandrill.png', isDefault: false }),
      ];

      element.open(images);
      await element.updateComplete;

      expect(element.hasAttribute('open')).toBe(true);
    });

    it('should display all provided images', async () => {
      const images = [
        new StaticImage({ id: 'lena', displayName: 'Lena', path: '/data/lena.png', isDefault: true }),
        new StaticImage({ id: 'mandrill', displayName: 'Mandrill', path: '/data/mandrill.png', isDefault: false }),
        new StaticImage({ id: 'peppers', displayName: 'Peppers', path: '/data/peppers.png', isDefault: false }),
      ];

      element.open(images);
      await element.updateComplete;

      const imageItems = element.shadowRoot!.querySelectorAll('.image-item');
      expect(imageItems.length).toBe(3);
    });

    it('should show empty state when no images provided', async () => {
      element.open([]);
      await element.updateComplete;

      const emptyState = element.shadowRoot!.querySelector('.empty-state');
      expect(emptyState).toBeDefined();
      expect(emptyState?.textContent).toContain('No images available');
    });
  });

  describe('Closing modal', () => {
    it('should close when close method is called', async () => {
      const images = [
        new StaticImage({ id: 'lena', displayName: 'Lena', path: '/data/lena.png', isDefault: true }),
      ];

      element.open(images);
      await element.updateComplete;
      expect(element.hasAttribute('open')).toBe(true);

      element.close();
      await element.updateComplete;
      expect(element.hasAttribute('open')).toBe(false);
    });

    it('should emit image-selected event when image is clicked', async () => {
      const images = [
        new StaticImage({ id: 'mandrill', displayName: 'Mandrill', path: '/data/mandrill.png', isDefault: false }),
      ];

      let eventFired = false;
      let selectedImage: StaticImage | null = null;

      element.addEventListener('image-selected', ((e: CustomEvent) => {
        eventFired = true;
        selectedImage = e.detail.image;
      }) as EventListener);

      element.open(images);
      await element.updateComplete;

      const imageItem = element.shadowRoot!.querySelector('[data-testid="image-item-mandrill"]') as HTMLElement;
      expect(imageItem).toBeDefined();

      imageItem.click();
      await element.updateComplete;

      expect(eventFired).toBe(true);
      expect(selectedImage?.id).toBe('mandrill');
      expect(element.hasAttribute('open')).toBe(false);
    });
  });

  describe('Default badge', () => {
    it('should show default badge on default image', async () => {
      const images = [
        new StaticImage({ id: 'lena', displayName: 'Lena', path: '/data/lena.png', isDefault: true }),
        new StaticImage({ id: 'mandrill', displayName: 'Mandrill', path: '/data/mandrill.png', isDefault: false }),
      ];

      element.open(images);
      await element.updateComplete;

      const lenaItem = element.shadowRoot!.querySelector('[data-testid="image-item-lena"]');
      const badge = lenaItem?.querySelector('.image-badge');
      expect(badge).toBeDefined();
      expect(badge?.textContent).toContain('Default');

      const mandrillItem = element.shadowRoot!.querySelector('[data-testid="image-item-mandrill"]');
      const noBadge = mandrillItem?.querySelector('.image-badge');
      expect(noBadge).toBeNull();
    });
  });
});

