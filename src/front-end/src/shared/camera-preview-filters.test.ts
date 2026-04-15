import { describe, expect, it } from 'vitest';
import { buildCameraPreviewFilterValue } from './camera-preview-filters';

describe('buildCameraPreviewFilterValue', () => {
  it('returns none when there are no active filters', () => {
    expect(buildCameraPreviewFilterValue([])).toBe('none');
    expect(buildCameraPreviewFilterValue(undefined)).toBe('none');
  });

  it('maps grayscale filter to css grayscale', () => {
    expect(
      buildCameraPreviewFilterValue([{ id: 'grayscale', parameters: { algorithm: 'bt601' } }])
    ).toBe('grayscale(1)');
  });

  it('maps blur filter using sigma when available', () => {
    expect(
      buildCameraPreviewFilterValue([{ id: 'blur', parameters: { sigma: '1.5' } }])
    ).toBe('blur(1.50px)');
  });

  it('composes grayscale and blur filters in order', () => {
    expect(
      buildCameraPreviewFilterValue([
        { id: 'grayscale', parameters: { algorithm: 'bt709' } },
        { id: 'blur', parameters: { kernel_size: '5' } },
      ])
    ).toBe('grayscale(1) blur(2.50px)');
  });
});
