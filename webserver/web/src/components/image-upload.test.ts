import { describe, it, expect, beforeEach } from 'vitest';
import './image-upload';
import type { ImageUpload } from './image-upload';

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

