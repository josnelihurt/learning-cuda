import { describe, it, expect } from 'vitest';
import { FilterData } from './FilterData';
import { FilterType } from '../../gen/common_pb';

describe('FilterData', () => {
  it('creates none filter by default', () => {
    const sut = new FilterData('none');
    expect(sut.getType()).toBe('none');
    expect(sut.toProtocol()).toBe(FilterType.NONE);
    expect(sut.hasParameters()).toBe(false);
  });

  it('creates grayscale filter and reports algorithm', () => {
    const sut = new FilterData('grayscale', { algorithm: 'bt709' });
    expect(sut.isGrayscale()).toBe(true);
    expect(sut.getParameter('algorithm')).toBe('bt709');
    expect(sut.toProtocol()).toBe(FilterType.GRAYSCALE);
  });

  it('accepts unknown filter types without throwing', () => {
    const sut = new FilterData('custom-filter', { strength: 'high' });
    expect(sut.getType()).toBe('custom-filter');
    expect(sut.toProtocol()).toBe(FilterType.UNSPECIFIED);
    expect(sut.getParameter('strength')).toBe('high');
  });

  it('validates blur kernel size', () => {
    expect(() => new FilterData('blur', { kernel_size: 4 })).toThrow('kernel_size must be a positive odd number');
    expect(() => new FilterData('blur', { kernel_size: 5 })).not.toThrow();
  });

  it('validates blur sigma', () => {
    expect(() => new FilterData('blur', { sigma: -1 })).toThrow('sigma must be a non-negative number');
    expect(() => new FilterData('blur', { sigma: 'invalid' })).toThrow('sigma must be a non-negative number');
    expect(() => new FilterData('blur', { sigma: 1.5 })).not.toThrow();
  });

  it('validates empty filter type', () => {
    expect(() => new FilterData('' as any)).toThrow('Filter type cannot be empty');
  });

  it('compares equality by type and parameters', () => {
    const a = new FilterData('none');
    const b = new FilterData('none');
    const c = new FilterData('grayscale');
    expect(a.equals(b)).toBe(true);
    expect(a.equals(c)).toBe(false);
  });

  it('exposes immutable parameters snapshot', () => {
    const sut = new FilterData('none', { radius: 5 });
    const params = sut.getParameters();
    params.radius = 10;
    expect(sut.getParameter('radius')).toBe(5);
  });
});
