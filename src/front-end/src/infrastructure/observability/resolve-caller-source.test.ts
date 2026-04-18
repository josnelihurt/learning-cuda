import { afterEach, describe, expect, it, vi } from 'vitest';
import { clearCallerSourceMapCache, resolveBundleSiteToSourceLabel } from './resolve-caller-source';

const minimalMap = {
  version: 3 as const,
  file: 'main-abc.js',
  sources: ['src/domain/example.ts'],
  names: [],
  mappings: 'AAAA',
};

afterEach(() => {
  clearCallerSourceMapCache();
});

describe('resolveBundleSiteToSourceLabel', () => {
  it('maps generated position through fetched source map', async () => {
    const fetchFn = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => minimalMap,
    });

    const label = await resolveBundleSiteToSourceLabel(
      'https://ex.com/assets/main-abc.js.map',
      1,
      0,
      fetchFn as unknown as typeof fetch
    );

    expect(fetchFn).toHaveBeenCalledWith('https://ex.com/assets/main-abc.js.map');
    expect(label).toBe('example.ts@1');
  });

  it('returns undefined when fetch fails', async () => {
    const fetchFn = vi.fn().mockRejectedValue(new Error('network'));

    const label = await resolveBundleSiteToSourceLabel(
      'https://ex.com/assets/x.js.map',
      1,
      0,
      fetchFn as unknown as typeof fetch
    );

    expect(label).toBeUndefined();
  });

  it('dedupes map fetches by URL', async () => {
    const fetchFn = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => minimalMap,
    });

    const url = 'https://ex.com/assets/main-abc.js.map';
    await resolveBundleSiteToSourceLabel(url, 1, 0, fetchFn as unknown as typeof fetch);
    await resolveBundleSiteToSourceLabel(url, 1, 0, fetchFn as unknown as typeof fetch);

    expect(fetchFn).toHaveBeenCalledTimes(1);
  });
});
