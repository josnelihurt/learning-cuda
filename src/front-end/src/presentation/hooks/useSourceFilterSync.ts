import { useEffect } from 'react';
import type { Dispatch, RefObject } from 'react';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import type { IToastDisplay } from '@/infrastructure/transport/transport-types';
import type { GridSource, GridSourceAction } from '@/presentation/utils/grid-source';
import { GridSourceActionType } from '@/presentation/utils/grid-source';
import { normalizeFilters } from '@/presentation/utils/grid-source';
import { webrtcService } from '@/infrastructure/connection/webrtc-service';
import { logger } from '@/infrastructure/observability/otel-logger';
import type { useCameraTransport } from '@/presentation/hooks/useCameraTransport';
import type { useFilterApplication } from '@/presentation/hooks/useFilterApplication';
import { AcceleratorType } from '@/gen/common_pb';

type SourceFilterSyncOptions = {
  ready: boolean;
  selectedSourceId: string | null;
  sourcesRef: RefObject<GridSource[]>;
  activeFilters: ActiveFilterState[];
  selectedAccelerator: AcceleratorType;
  selectedResolution: string;
  dispatch: Dispatch<GridSourceAction>;
  sendCameraControlRequest: ReturnType<typeof useCameraTransport>['sendCameraControlRequest'];
  applyStaticFilters: ReturnType<typeof useFilterApplication>['applyStaticFilters'];
  applyVideoFilters: ReturnType<typeof useFilterApplication>['applyVideoFilters'];
  toastManager: IToastDisplay;
};

export function useSourceFilterSync({
  ready,
  selectedSourceId,
  sourcesRef,
  activeFilters,
  selectedAccelerator,
  selectedResolution,
  dispatch,
  sendCameraControlRequest,
  applyStaticFilters,
  applyVideoFilters,
  toastManager,
}: SourceFilterSyncOptions): void {
  useEffect(() => {
    if (!ready || !selectedSourceId) return;
    const selectedSource = sourcesRef.current.find((s) => s.id === selectedSourceId);
    if (!selectedSource) return;

    const normalizedFilters = normalizeFilters(activeFilters);
    const hasDetectionFilter = normalizedFilters.some((filter) => filter.id === 'model_inference');

    dispatch({
      type: GridSourceActionType.SYNC_FILTERS,
      payload: {
        sourceId: selectedSource.id,
        filters: normalizedFilters,
        resolution: selectedResolution,
        accelerator: selectedAccelerator,
      },
    });

    if (!hasDetectionFilter) {
      dispatch({
        type: GridSourceActionType.SET_DETECTIONS,
        payload: { sourceId: selectedSource.id, detections: [], width: 0, height: 0 },
      });
    }

    if (selectedSource.type === 'video') {
      void applyVideoFilters({
        source: selectedSource,
        filters: normalizedFilters,
        accelerator: selectedAccelerator,
        resolution: selectedResolution,
        onSourceUpdate: (sourceId, updater) =>
          dispatch({ type: GridSourceActionType.UPDATE_SOURCE, payload: { sourceId, updater } }),
      }).then((result) => {
        if (result.status === 'error') {
          toastManager.error('Filter update failed', result.message);
          logger.error('Video filter update failed', { 'source.id': selectedSource.id });
        }
      });
      return;
    }

    if (selectedSource.type === 'camera') {
      if (!selectedSource.sessionId) {
        logger.warn('Camera filter update skipped - no sessionId', {
          'source.id': selectedSource.id,
          'source.name': selectedSource.name,
        });
        return;
      }
      if (!webrtcService.isDataChannelOpen(selectedSource.sessionId)) {
        logger.warn('Camera filter update skipped - data channel not open', {
          'source.id': selectedSource.id,
          'source.sessionId': selectedSource.sessionId,
        });
        return;
      }
      sendCameraControlRequest(
        selectedSource.sessionId,
        selectedSource.id,
        normalizedFilters,
        selectedAccelerator
      );
      return;
    }

    void applyStaticFilters({
      source: selectedSource,
      filters: normalizedFilters,
      accelerator: selectedAccelerator,
      resolution: selectedResolution,
      onSourceUpdate: (sourceId, updater) =>
        dispatch({ type: GridSourceActionType.UPDATE_SOURCE, payload: { sourceId, updater } }),
    }).then((result) => {
      if (result.status === 'error') {
        toastManager.error('Static image processing failed', result.message);
      }
    });
  }, [
    activeFilters,
    applyStaticFilters,
    applyVideoFilters,
    dispatch,
    ready,
    selectedAccelerator,
    selectedResolution,
    selectedSourceId,
    sendCameraControlRequest,
    sourcesRef,
    toastManager,
  ]);
}
