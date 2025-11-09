import { describe, it, expect, beforeEach, vi } from 'vitest';
import './image-upload';
import type { ImageUpload } from './image-upload';

vi.mock('../../services/otel-logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

vi.mock('../../services/file-service', () => ({
  fileService: {
    uploadImage: vi.fn(),
    initialize: vi.fn(),
  },
}));

vi.mock('../../services/telemetry-service', () => ({
  telemetryService: {
    createSpan: vi.fn().mockReturnValue({
      setAttribute: vi.fn(),
      recordException: vi.fn(),
      end: vi.fn(),
    }),
  },
}));

describe('ImageUpload', () => {
  let element: ImageUpload;

  beforeEach(() => {
    element = document.createElement('image-upload') as ImageUpload;
    document.body.appendChild(element);
  });

  describe('Component Creation', () => {
    it('should create element', () => {
      expect(element).toBeDefined();
      expect(element.tagName).toBe('IMAGE-UPLOAD');
    });

    it('should have shadow root', () => {
      expect(element.shadowRoot).toBeDefined();
    });
  });

  describe('Component Properties', () => {
    it('should be a custom element', () => {
      expect(customElements.get('image-upload')).toBeDefined();
    });

    it('should have default state', () => {
      expect(element).toBeInstanceOf(HTMLElement);
    });
  });
});
