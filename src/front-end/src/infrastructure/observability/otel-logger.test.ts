import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const parseCallerFromStackMock = vi.hoisted(() => vi.fn());

vi.mock('./caller-site', () => ({
  parseCallerFromStack: (...args: unknown[]) => parseCallerFromStackMock(...args),
}));

vi.mock('./telemetry-service', () => ({
  telemetryService: {
    getTraceHeaders: vi.fn(() => ({})),
  },
}));

import { clearCallerSourceMapCache } from './resolve-caller-source';
import { OtelLogger } from './otel-logger';

describe('OtelLogger console caller', () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
    window.location.href = 'https://app.example/';
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    clearCallerSourceMapCache();
    vi.restoreAllMocks();
    parseCallerFromStackMock.mockReset();
  });

  it('prints resolved src path for bundle frames after source map loads', async () => {
    const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        version: 3 as const,
        file: 'main-abc.js',
        sources: ['src/infrastructure/foo.ts'],
        names: [],
        mappings: 'AAAA',
      }),
    }) as unknown as typeof fetch;

    parseCallerFromStackMock.mockReturnValue({
      kind: 'bundle',
      label: 'main-abc.js@1:0',
      mapUrl: 'https://app.example/assets/main-abc.js.map',
      line: 1,
      column: 0,
    });

    const log = new OtelLogger();
    log.initialize('DEBUG', true, 'production', false);
    log.info('hello');

    await vi.waitUntil(() => infoSpy.mock.calls.length > 0, { timeout: 2000 });

    const first = String(infoSpy.mock.calls[0]?.[0] ?? '');
    expect(first).toContain('[src/infrastructure/foo.ts@1]');
    expect(first).toContain('hello');
  });

  it('falls back to bundle label when source map is missing', async () => {
    const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
    }) as unknown as typeof fetch;

    parseCallerFromStackMock.mockReturnValue({
      kind: 'bundle',
      label: 'main-abc.js@9:1',
      mapUrl: 'https://app.example/assets/main-abc.js.map',
      line: 9,
      column: 1,
    });

    const log = new OtelLogger();
    log.initialize('DEBUG', true, 'production', false);
    log.info('hello');

    await vi.waitUntil(() => infoSpy.mock.calls.length > 0, { timeout: 2000 });

    const first = String(infoSpy.mock.calls[0]?.[0] ?? '');
    expect(first).toBe('[main-abc.js@9:1] hello');
  });

  it('prints sync caller label for non-bundle frames', () => {
    const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});

    parseCallerFromStackMock.mockReturnValue({
      kind: 'source',
      label: 'webrtc-service.ts@42',
    });

    const log = new OtelLogger();
    log.initialize('DEBUG', true, 'production', false);
    log.info('x');

    const first = String(infoSpy.mock.calls[0]?.[0] ?? '');
    expect(first).toBe('[webrtc-service.ts@42] x');
  });
});
