import { useCallback, useRef, useState } from 'react';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import type { WebRTCFrameTransportService } from '@/infrastructure/transport/webrtc-frame-transport';
import { FilterData } from '@/domain/value-objects';
import { logger } from '@/infrastructure/observability/otel-logger';
import { AcceleratorType } from '@/gen/common_pb';

export type VideoFilterState =
  | { status: 'idle' }
  | { status: 'stopping'; pendingFilters: ActiveFilterState[] }
  | { status: 'starting'; pendingFilters: ActiveFilterState[] }
  | { status: 'error'; error: string };

export interface VideoFilterManagerOptions {
  transport: WebRTCFrameTransportService | null;
  videoId: string;
  onError?: (message: string) => void;
}

const VIDEO_RESTART_DELAY_MS = 200;
const MAX_RETRIES = 2;

export function useVideoFilterManager() {
  const [state, setState] = useState<VideoFilterState>({ status: 'idle' });
  const pendingUpdateRef = useRef<{ filters: ActiveFilterState[]; accelerator: AcceleratorType; retryCount: number } | null>(null);

  const updateFilters = useCallback(
    async (options: VideoFilterManagerOptions & { filters: ActiveFilterState[]; accelerator: AcceleratorType }) => {
      const { transport, videoId, filters, accelerator, onError } = options;

      if (!transport || !transport.isConnected()) {
        const errorMsg = 'Frame transport not connected';
        logger.error('Frame transport not connected for video filter update', {
          'video.id': videoId,
        });
        setState({ status: 'error', error: errorMsg });
        onError?.(errorMsg);
        return { status: 'error' } as const;
      }

      pendingUpdateRef.current = { filters, accelerator, retryCount: 0 };
      setState({ status: 'stopping', pendingFilters: filters });

      try {
        transport.sendStopVideo(videoId);

        await new Promise<void>((resolve) => {
          const timeoutId = setTimeout(() => {
            const pending = pendingUpdateRef.current;
            if (!pending) {
              resolve();
              return;
            }

            if (transport?.isConnected()) {
              const filterData = pending.filters.map(
                (f) => new FilterData(f.id, { ...f.parameters })
              );
              transport.sendStartVideo(videoId, filterData, pending.accelerator);
              setState({ status: 'idle' });
              pendingUpdateRef.current = null;
              resolve();
            } else {
              const retryCount = pending.retryCount + 1;
              if (retryCount <= MAX_RETRIES) {
                pendingUpdateRef.current = { ...pending, retryCount };
                logger.warn('Video transport not ready, retrying...', {
                  'video.id': videoId,
                  'retry.count': retryCount,
                });
                setTimeout(() => {
                  if (transport?.isConnected()) {
                    const filterData = pending.filters.map(
                      (f) => new FilterData(f.id, { ...f.parameters })
                    );
                    transport.sendStartVideo(videoId, filterData, pending.accelerator);
                    setState({ status: 'idle' });
                    pendingUpdateRef.current = null;
                  } else {
                    const errorMsg = 'Could not restart video stream after retry';
                    setState({ status: 'error', error: errorMsg });
                    onError?.(errorMsg);
                  }
                  resolve();
                }, VIDEO_RESTART_DELAY_MS);
              } else {
                const errorMsg = 'Could not restart video stream. Please try selecting the video again.';
                setState({ status: 'error', error: errorMsg });
                onError?.(errorMsg);
                logger.error('Video filter update failed - max retries exceeded', {
                  'video.id': videoId,
                });
                resolve();
              }
            }
          }, VIDEO_RESTART_DELAY_MS);

          return () => clearTimeout(timeoutId);
        });

        return { status: 'success' } as const;
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error updating video filters';
        setState({ status: 'error', error: errorMsg });
        onError?.(errorMsg);
        logger.error('Exception during video filter update', {
          'error.message': errorMsg,
          'video.id': videoId,
        });
        return { status: 'error' } as const;
      }
    },
    []
  );

  const reset = useCallback(() => {
    setState({ status: 'idle' });
    pendingUpdateRef.current = null;
  }, []);

  return { state, updateFilters, reset };
}
