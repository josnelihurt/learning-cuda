import { describe, expect, it } from 'vitest';
import { jsUrlToMapUrl, parseCallerFromStack } from './caller-site';

describe('jsUrlToMapUrl', () => {
  it('replaces trailing .js with .js.map', () => {
    expect(jsUrlToMapUrl('https://ex.com/assets/main-abc123.js')).toBe(
      'https://ex.com/assets/main-abc123.js.map'
    );
  });
});

describe('parseCallerFromStack', () => {
  const origin = 'https://ex.com';

  it('parses absolute same-origin hashed chunk URL as bundle', () => {
    const stack = `Error: x
    at foo (https://ex.com/assets/main-2TwQI6dV.js:50:12)
    at otel-logger.ts:1:1`;
    const r = parseCallerFromStack(stack, origin);
    expect(r?.kind).toBe('bundle');
    if (r?.kind === 'bundle') {
      expect(r.label).toBe('main-2TwQI6dV.js@50:12');
      expect(r.mapUrl).toBe('https://ex.com/assets/main-2TwQI6dV.js.map');
      expect(r.line).toBe(50);
      expect(r.column).toBe(12);
    }
  });

  it('parses /assets/ path in frame as bundle', () => {
    const stack = `    at t (/assets/index-abc12345.js:3:0)`;
    const r = parseCallerFromStack(stack, origin);
    expect(r?.kind).toBe('bundle');
    if (r?.kind === 'bundle') {
      expect(r.mapUrl).toBe('https://ex.com/assets/index-abc12345.js.map');
    }
  });

  it('parses TypeScript source frame as source', () => {
    const stack = `    at connect (webrtc-service.ts:120:4)`;
    const r = parseCallerFromStack(stack, origin);
    expect(r).toEqual({ kind: 'source', label: 'webrtc-service.ts@120' });
  });

  it('skips otel-logger frames', () => {
    const stack = `    at OtelLogger.log (/src/infrastructure/observability/otel-logger.ts:10:0)
    at bar (https://ex.com/assets/main-xx.js:50:1)`;
    const r = parseCallerFromStack(stack, origin);
    expect(r?.kind).toBe('bundle');
    if (r?.kind === 'bundle') {
      expect(r.line).toBe(50);
    }
  });

  it('does not treat cross-origin .js as bundle', () => {
    const stack = `    at x (https://cdn.example.com/lib.js:1:0)`;
    expect(parseCallerFromStack(stack, origin)).toBeUndefined();
  });

  it('parses basename-only hashed chunk with /assets/ convention', () => {
    const stack = `    at n (main-2TwQI6dV.js:9:0)`;
    const r = parseCallerFromStack(stack, origin);
    expect(r?.kind).toBe('bundle');
    if (r?.kind === 'bundle') {
      expect(r.mapUrl).toBe('https://ex.com/assets/main-2TwQI6dV.js.map');
    }
  });
});
