import { describe, it, expect } from 'vitest';
import { FilterData } from './filter-data';
import { FilterType } from '@/gen/common_pb';

describe('FilterData', () => {
  it('creates arbitrary filter without hardcoded semantics', () => {
    const sut = new FilterData('custom_edge_enhance');
    expect(sut.getType()).toBe('custom_edge_enhance');
    expect(sut.toProtocol()).toBe(FilterType.UNSPECIFIED);
    expect(sut.hasParameters()).toBe(false);
  });

  it('stores generic parameter maps as-is', () => {
    const sut = new FilterData('arbitrary_filter_v2', { alpha: '0.7', enabled: true });
    expect(sut.getParameter('alpha')).toBe('0.7');
    expect(sut.getParameter('enabled')).toBe(true);
  });

  it('accepts unknown filter types without throwing', () => {
    const sut = new FilterData('custom-filter', { strength: 'high' });
    expect(sut.getType()).toBe('custom-filter');
    expect(sut.toProtocol()).toBe(FilterType.UNSPECIFIED);
    expect(sut.getParameter('strength')).toBe('high');
  });

  it('validates parameters shape only', () => {
    expect(() => new FilterData('custom', null as any)).toThrow('Filter parameters must be an object');
    expect(() => new FilterData('custom', [] as any)).toThrow('Filter parameters must be an object');
    expect(() => new FilterData('custom', { any: 'value' })).not.toThrow();
  });

  it('validates empty filter type', () => {
    expect(() => new FilterData('' as any)).toThrow('Filter type cannot be empty');
  });

  it('compares equality by type and parameters', () => {
    const a = new FilterData('f_a');
    const b = new FilterData('f_a');
    const c = new FilterData('f_b');
    expect(a.equals(b)).toBe(true);
    expect(a.equals(c)).toBe(false);
  });

  it('exposes immutable parameters snapshot', () => {
    const sut = new FilterData('x_filter', { radius: 5 });
    const params = sut.getParameters();
    params.radius = 10;
    expect(sut.getParameter('radius')).toBe(5);
  });
});
