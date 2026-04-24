import { TraceMap, originalPositionFor, type EncodedSourceMap } from '@jridgewell/trace-mapping';

const traceMapCache = new Map<string, Promise<TraceMap | null>>();

export function clearCallerSourceMapCache(): void {
  traceMapCache.clear();
}

async function loadTraceMap(mapUrl: string, fetchFn: typeof fetch): Promise<TraceMap | null> {
  let pending = traceMapCache.get(mapUrl);
  if (!pending) {
    pending = (async () => {
      try {
        const res = await fetchFn(mapUrl);
        if (!res.ok) {
          return null;
        }
        const json: unknown = await res.json();
        return new TraceMap(json as EncodedSourceMap, mapUrl);
      } catch (e) {
        console.debug('Failed to load source map:', mapUrl, e);
        return null;
      }
    })();
    traceMapCache.set(mapUrl, pending);
  }
  return pending;
}

/** Prefer `src/...` tail when present so labels disambiguate same basenames. */
export function sourcePathForLabel(source: string): string {
  const norm = source.replace(/\\/g, '/');
  const parts = norm.split('/').filter((p) => p.length > 0);
  const srcIdx = parts.lastIndexOf('src');
  if (srcIdx >= 0) {
    return parts.slice(srcIdx).join('/');
  }
  return parts[parts.length - 1] || norm;
}

export async function resolveBundleSiteToSourceLabel(
  mapUrl: string,
  line: number,
  column: number,
  fetchFn: typeof fetch = fetch
): Promise<string | undefined> {
  const map = await loadTraceMap(mapUrl, fetchFn);
  if (!map) {
    return undefined;
  }
  const pos = originalPositionFor(map, { line, column });
  if (pos.source == null || pos.line == null) {
    return undefined;
  }
  const path = sourcePathForLabel(pos.source);
  return `${path}@${pos.line}`;
}
