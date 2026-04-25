import { useRef } from 'react';
import type { RefObject } from 'react';

/**
 * Keeps a ref aligned with the latest value on every render, without `useEffect`.
 *
 * Use when effects, intervals, animation frames, or promise continuations read a
 * callback or props object and must not capture a stale closure. This replaces
 * repetitive patterns such as `useRef(x)` plus `useEffect(() => { ref.current = x }, [x])`.
 *
 * The update runs synchronously during render, so subscribers that run later in the
 * same tick (or after layout/paint) always see the value from the most recent render.
 */
export function useLatest<T>(value: T): RefObject<T> {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}
