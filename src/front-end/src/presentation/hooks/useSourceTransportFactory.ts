import { useCallback, useRef } from 'react';
import type { Dispatch, RefObject } from 'react';
import type { InputSource } from '@/gen/config_service_pb';
import type { IStatsDisplay, ICameraPreview, IToastDisplay } from '@/infrastructure/transport/transport-types';
import { WebRTCFrameTransportService } from '@/infrastructure/transport/webrtc-frame-transport';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import type { GridSource, GridSourceAction } from '@/presentation/components/video/grid-source';
import { GridSourceActionType } from '@/presentation/components/video/grid-source';
import { filtersToFilterData } from '@/presentation/components/video/grid-source';
import { frameResponseToDataUrl } from '@/presentation/utils/image-utils';

type SourceTransportFactoryOptions = {
  nextNumberRef: RefObject<number>;
  statsManager: Pick<IStatsDisplay, 'updateCameraStatus' | 'updateTransportStatus' | 'updateProcessingStats'>;
  toastManager: IToastDisplay;
  dispatch: Dispatch<GridSourceAction>;
  activeFiltersRef: RefObject<ActiveFilterState[]>;
  selectedResolutionRef: RefObject<string>;
  selectedAcceleratorRef: RefObject<'gpu' | 'cpu'>;
};

type SourceTransportFactoryResult = {
  buildSource: (inputSource: InputSource) => GridSource;
};

export function useSourceTransportFactory({
  nextNumberRef,
  statsManager,
  toastManager,
  dispatch,
  activeFiltersRef,
  selectedResolutionRef,
  selectedAcceleratorRef,
}: SourceTransportFactoryOptions): SourceTransportFactoryResult {
  const cameraFrameTimeRef = useRef<Record<string, number>>({});

  const buildSource = useCallback(
    (inputSource: InputSource): GridSource => {
      const number = nextNumberRef.current;
      nextNumberRef.current += 1;

      const uniqueId = `${inputSource.id}-${number}`;
      const sourceImagePath =
        inputSource.type === 'video'
          ? inputSource.previewImagePath || ''
          : inputSource.imagePath;

      const activeFilters = activeFiltersRef.current;
      const selectedResolution = selectedResolutionRef.current;
      const selectedAccelerator = selectedAcceleratorRef.current;

      const cameraManager: ICameraPreview = {
        setProcessing: () => undefined,
        getLastFrameTime: () => cameraFrameTimeRef.current[uniqueId] ?? performance.now(),
      };

      const transport =
        inputSource.type === 'camera'
          ? null
          : new WebRTCFrameTransportService(
              uniqueId,
              statsManager as IStatsDisplay,
              cameraManager,
              toastManager
            );

      if (transport) {
        transport.connect();
        transport.onFrameResult((data) => {
          if (!data.imageData?.byteLength || data.width <= 0 || data.height <= 0) return;
          dispatch({
            type: GridSourceActionType.SET_CURRENT_IMAGE,
            payload: {
              sourceId: uniqueId,
              currentImageSrc: frameResponseToDataUrl(
                data.imageData,
                data.width,
                data.height,
                data.channels || 4
              ),
            },
          });
        });
        transport.onDetectionResult((frame) => {
          dispatch({
            type: GridSourceActionType.SET_DETECTIONS,
            payload: {
              sourceId: uniqueId,
              detections: frame.detections,
              width: frame.imageWidth,
              height: frame.imageHeight,
            },
          });
        });

        const initialFilters =
          activeFilters.length > 0
            ? activeFilters.map((f) => ({ id: f.id, parameters: { ...f.parameters } }))
            : [{ id: 'none', parameters: {} }];

        if (inputSource.type === 'video') {
          waitForConnected(transport, () => {
            dispatch({ type: GridSourceActionType.SET_CONNECTED, payload: { sourceId: uniqueId, connected: true } });
            transport.sendStartVideo(inputSource.id, filtersToFilterData(initialFilters), selectedAccelerator);
          });
        } else {
          waitForConnected(transport, () => {
            dispatch({ type: GridSourceActionType.SET_CONNECTED, payload: { sourceId: uniqueId, connected: true } });
          });
        }
      }

      const filters =
        activeFilters.length > 0
          ? activeFilters.map((f) => ({ id: f.id, parameters: { ...f.parameters } }))
          : [{ id: 'none', parameters: {} }];

      return {
        id: uniqueId,
        number,
        name: inputSource.displayName,
        type: inputSource.type,
        imagePath: sourceImagePath,
        originalImageSrc: sourceImagePath,
        currentImageSrc: sourceImagePath,
        transport,
        remoteStream: null,
        sessionId: null,
        sessionMode: inputSource.type === 'camera' ? 'camera-mediatrack' : 'frame-processing',
        filters,
        resolution: selectedResolution || 'original',
        accelerator: selectedAccelerator || 'gpu',
        videoId: inputSource.type === 'video' ? inputSource.id : undefined,
        detections: [],
        detectionImageWidth: 0,
        detectionImageHeight: 0,
        connected: false,
      };
    },
    [activeFiltersRef, dispatch, nextNumberRef, selectedAcceleratorRef, selectedResolutionRef, statsManager, toastManager]
  );

  return { buildSource };
}

function waitForConnected(
  transport: WebRTCFrameTransportService,
  onConnected: () => void
): void {
  const check = (): void => {
    if (transport.isConnected()) {
      onConnected();
    } else {
      setTimeout(check, 100);
    }
  };
  setTimeout(check, 100);
}
