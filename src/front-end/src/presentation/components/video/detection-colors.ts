const DETECTION_PALETTE = [
  '#ef4444',
  '#f59e0b',
  '#10b981',
  '#3b82f6',
  '#8b5cf6',
  '#ec4899',
  '#14b8a6',
  '#f97316',
  '#84cc16',
  '#06b6d4',
  '#a855f7',
  '#e11d48',
];

export function colorForClassId(classId: number): string {
  const normalized = Number.isFinite(classId) ? Math.abs(Math.trunc(classId)) : 0;
  return DETECTION_PALETTE[normalized % DETECTION_PALETTE.length];
}
