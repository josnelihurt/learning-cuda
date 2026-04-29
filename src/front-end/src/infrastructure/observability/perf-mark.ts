import { logger } from '@/infrastructure/observability/otel-logger';

const PREFIX = 'cuda-learning';

const measurements: Map<string, number> = new Map();
let markCounter = 0;

function uniqueName(label: string): string {
  markCounter += 1;
  return `${PREFIX}:${label}:${markCounter}`;
}

export function markStart(label: string): string {
  const name = uniqueName(label);
  try {
    performance.mark(name);
  } catch {
    void name;
  }
  return name;
}

export function markEnd(label: string, startMark: string): number {
  const endName = uniqueName(label);
  const measureName = `${PREFIX}.measure:${label}:${markCounter}`;
  try {
    performance.mark(endName);
    performance.measure(measureName, startMark, endName);
    const entries = performance.getEntriesByName(measureName);
    const duration = entries.length > 0 ? entries[entries.length - 1].duration : 0;
    const rounded = Math.round(duration * 100) / 100;
    measurements.set(label, rounded);
    logger.info(`[perf] ${label} ${rounded.toFixed(1)}ms`);
    return rounded;
  } catch {
    return -1;
  }
}

export function getMeasurements(): ReadonlyMap<string, number> {
  return measurements;
}

export function logTimingSummary(section: string, prefixes: string[]): void {
  const relevant: Array<[string, number]> = [];
  for (const [label, duration] of measurements) {
    if (prefixes.some((p) => label.startsWith(p))) {
      relevant.push([label, duration]);
    }
  }
  if (relevant.length === 0) {
    return;
  }
  const maxLabelLen = Math.max(...relevant.map(([l]) => l.length));
  const lines = relevant.map(
    ([label, dur]) => `  ${label.padEnd(maxLabelLen + 2)} ${dur.toFixed(1)}ms`
  );
  logger.info(`\n[${section}]\n${lines.join('\n')}`);
}
