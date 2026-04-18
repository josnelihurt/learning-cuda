const OTEL_LOGGER_SKIP = /[/\\]otel-logger\.(t|j)sx?\b/;

export type ParsedCaller =
  | { kind: 'source'; label: string }
  | {
      kind: 'bundle';
      label: string;
      mapUrl: string;
      line: number;
      column: number;
    };

export function jsUrlToMapUrl(jsUrl: string): string {
  const u = new URL(jsUrl);
  if (/\.js$/i.test(u.pathname)) {
    u.pathname = u.pathname.replace(/\.js$/i, '.js.map');
  } else {
    u.pathname += '.map';
  }
  return u.toString();
}

function sameOriginBundleJs(jsUrl: string, pageOrigin: string): boolean {
  try {
    const u = new URL(jsUrl);
    const o = new URL(pageOrigin);
    if (u.origin !== o.origin || !u.pathname.toLowerCase().endsWith('.js')) {
      return false;
    }
    if (u.pathname.includes('node_modules')) {
      return false;
    }
    const p = u.pathname;
    return p.includes('/assets/') || /-[A-Za-z0-9]{6,}\.js$/i.test(p);
  } catch {
    return false;
  }
}

function bundleFromAbsoluteJsUrl(jsUrl: string, line: number, column: number): ParsedCaller {
  const path = new URL(jsUrl).pathname;
  const base = path.split('/').pop() || path;
  return {
    kind: 'bundle',
    label: `${base}@${line}:${column}`,
    mapUrl: jsUrlToMapUrl(jsUrl),
    line,
    column,
  };
}

/**
 * Parses the first useful stack frame for logging: either a bundled chunk (needs source map)
 * or a source file path (e.g. Vite dev).
 */
export function parseCallerFromStack(stack: string, pageOrigin: string): ParsedCaller | undefined {
  const origin = pageOrigin.replace(/\/$/, '');

  for (const raw of stack.split('\n')) {
    const line = raw.trim();
    if (OTEL_LOGGER_SKIP.test(line)) continue;

    const absJs = line.match(/(https?:\/\/[^\s):]+\.js):(\d+):(\d+)/);
    if (absJs) {
      const jsUrl = absJs[1];
      const ln = Number(absJs[2]);
      const col = Number(absJs[3]);
      if (Number.isFinite(ln) && Number.isFinite(col) && sameOriginBundleJs(jsUrl, pageOrigin)) {
        return bundleFromAbsoluteJsUrl(jsUrl, ln, col);
      }
    }

    const assetsPath = line.match(/(\/assets\/[^\s):]+\.js):(\d+):(\d+)/);
    if (assetsPath) {
      const pathOnly = assetsPath[1];
      const ln = Number(assetsPath[2]);
      const col = Number(assetsPath[3]);
      if (Number.isFinite(ln) && Number.isFinite(col)) {
        const jsUrl = `${origin}${pathOnly}`;
        return bundleFromAbsoluteJsUrl(jsUrl, ln, col);
      }
    }

    const fileMatches = [
      ...line.matchAll(/\b([A-Za-z0-9_.-]+\.(?:tsx?|jsx?|mjs|cjs)):(\d+):(\d+)/g),
    ];
    if (fileMatches.length === 0) continue;

    const m = fileMatches[fileMatches.length - 1];
    const file = m[1];
    const ln = Number(m[2]);
    const col = Number(m[3]);
    if (!Number.isFinite(ln) || !Number.isFinite(col)) continue;

    if (file.endsWith('.js')) {
      if (line.includes('/assets/')) {
        const pathMatch = line.match(/(\/assets\/[^)\s]+\.js):(\d+):(\d+)/);
        if (pathMatch) {
          const jsUrl = `${origin}${pathMatch[1]}`;
          return bundleFromAbsoluteJsUrl(jsUrl, Number(pathMatch[2]), Number(pathMatch[3]));
        }
      }
      const looksLikeHashedChunk = /^[\w.-]+-[A-Za-z0-9]{6,}\.js$/.test(file);
      if (looksLikeHashedChunk) {
        const jsUrl = `${origin}/assets/${file}`;
        return {
          kind: 'bundle',
          label: `${file}@${ln}:${col}`,
          mapUrl: jsUrlToMapUrl(jsUrl),
          line: ln,
          column: col,
        };
      }
      continue;
    }

    return { kind: 'source', label: `${file}@${ln}` };
  }

  return undefined;
}
