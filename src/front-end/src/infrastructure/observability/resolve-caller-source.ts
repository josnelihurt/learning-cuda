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
      } catch {
        return null;
      }
    })();
    traceMapCache.set(mapUrl, pending);
  }
  return pending;
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
  const file = pos.source.split('/').pop() || pos.source;
  return `${file}@${pos.line}`;
}
