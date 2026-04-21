import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { InputSource } from '@/gen/config_service_pb';
import type { StaticImage } from '@/gen/common_pb';
import { AcceleratorType } from '@/gen/common_pb';
import type { IStatsDisplay, IToastDisplay } from '@/infrastructure/transport/transport-types';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import { useAppServices } from '@/presentation/providers/app-services-provider';
import { useDashboardState } from '@/presentation/context/dashboard-state-context';
import { WebRTCFrameTransportService } from '@/infrastructure/transport/webrtc-frame-transport';
import { webrtcService } from '@/infrastructure/connection/webrtc-service';
import { AcceleratorConfig, FilterData, GrayscaleAlgorithm } from '@/domain/value-objects';
import { logger } from '@/infrastructure/observability/otel-logger';
import { VideoGrid } from './VideoGrid';
import { SourceDrawer } from './SourceDrawer';
import { AddSourceFab } from './AddSourceFab';
import { ImageSelectorModal } from './ImageSelectorModal';
import { AcceleratorStatusFab } from './AcceleratorStatusFab';
import { useToast } from '@/presentation/hooks/useToast';
import { StatsPanel as ReactStatsPanel } from '@/presentation/components/app/StatsPanel';
import {
  type Detection,
  GenericFilterParameterSelection,
  GenericFilterSelection,
  ProcessImageRequest,
} from '@/gen/image_processor_service_pb';

type GridSource = {
  id: string;
  number: number;
  name: string;
  type: string;
  imagePath: string;
  originalImageSrc: string;
  currentImageSrc: string;
  transport: WebRTCFrameTransportService | null;
  remoteStream: MediaStream | null;
  sessionId: string | null;
  sessionMode: 'frame-processing' | 'camera-mediatrack';
  filters: ActiveFilterState[];
  resolution: string;
  accelerator: 'gpu' | 'cpu';
  videoId?: string;
  detections: Detection[];
  detectionImageWidth: number;
  detectionImageHeight: number;
};

const MAX_SOURCES = 9;

function toProtocolAccelerator(value: 'gpu' | 'cpu'): AcceleratorType {
  return value === 'cpu' ? AcceleratorType.CPU : AcceleratorType.CUDA;
}

function frameResponseToDataUrl(bytes: Uint8Array, width: number, height: number, channels: number): string {
  const rgba = new Uint8ClampedArray(width * height * 4);

  if (channels === 1) {
    for (let index = 0; index < width * height; index += 1) {
      const value = bytes[index] ?? 0;
      const offset = index * 4;
      rgba[offset] = value;
      rgba[offset + 1] = value;
      rgba[offset + 2] = value;
      rgba[offset + 3] = 255;
    }
  } else if (channels === 3) {
    for (let index = 0; index < width * height; index += 1) {
      const srcOffset = index * 3;
      const dstOffset = index * 4;
      rgba[dstOffset] = bytes[srcOffset] ?? 0;
      rgba[dstOffset + 1] = bytes[srcOffset + 1] ?? 0;
      rgba[dstOffset + 2] = bytes[srcOffset + 2] ?? 0;
      rgba[dstOffset + 3] = 255;
    }
  } else {
    for (let index = 0; index < width * height; index += 1) {
      const srcOffset = index * channels;
      const dstOffset = index * 4;
      rgba[dstOffset] = bytes[srcOffset] ?? 0;
      rgba[dstOffset + 1] = bytes[srcOffset + 1] ?? 0;
      rgba[dstOffset + 2] = bytes[srcOffset + 2] ?? 0;
      rgba[dstOffset + 3] = channels >= 4 ? (bytes[srcOffset + 3] ?? 255) : 255;
    }
  }

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas context is not available');
  }

  context.putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas.toDataURL('image/png');
}

async function rasterizeImageToRgb(
  imageSrc: string,
  width: number,
  height: number
): Promise<Uint8Array> {
  const image = new Image();
  image.crossOrigin = 'anonymous';

  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error('Failed to load original image'));
    image.src = imageSrc;
  });

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Canvas context is not available');
  }

  context.drawImage(image, 0, 0, width, height);
  const raster = context.getImageData(0, 0, width, height);
  const rgb = new Uint8Array(width * height * 3);

  for (let index = 0, pixel = 0; index < raster.data.length; index += 4, pixel += 3) {
    rgb[pixel] = raster.data[index];
    rgb[pixel + 1] = raster.data[index + 1];
    rgb[pixel + 2] = raster.data[index + 2];
  }

  return rgb;
}

export function VideoGridHost() {
  const [sources, setSources] = useState<GridSource[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isImageSelectorOpen, setIsImageSelectorOpen] = useState(false);
  const [availableSources, setAvailableSources] = useState<InputSource[]>([]);
  const [availableImages, setAvailableImages] = useState<StaticImage[]>([]);
  const [fps, setFps] = useState('--');
  const [time, setTime] = useState('--ms');
  const [frames, setFrames] = useState(0);
  const [cameraStatus, setCameraStatus] = useState('Inactive');
  const [cameraStatusType, setCameraStatusType] = useState<'success' | 'error' | 'warning' | 'inactive'>(
    'inactive'
  );
  const fpsHistoryRef = useRef<number[]>([]);
  const processingTimesRef = useRef<number[]>([]);
  const nextNumberRef = useRef(1);
  const defaultSourceInitializedRef = useRef(false);
  const cameraFrameTimeRef = useRef<Record<string, number>>({});
  const cameraSessionSourceIdsRef = useRef<Set<string>>(new Set());
  const pendingSourceNumberForImageChangeRef = useRef<number | null>(null);
  const sourcesRef = useRef<GridSource[]>([]);
  const { container, ready } = useAppServices();
  const toast = useToast();
  const {
    activeFilters,
    selectedAccelerator,
    selectedResolution,
    setSelectedSource,
    setActiveFilters,
    setResolution,
  } = useDashboardState();

  useEffect(() => {
    sourcesRef.current = sources;
  }, [sources]);

  const statsManager = useMemo(
    () =>
      ({
        updateCameraStatus: (status: string, type: 'success' | 'error' | 'warning' | 'inactive') => {
          setCameraStatus(status);
          setCameraStatusType(type);
        },
        updateTransportStatus: () => undefined,
        updateProcessingStats: (processingTime: number) => {
          setFrames((current) => current + 1);
          const instantFps = 1000 / processingTime;
          fpsHistoryRef.current.push(instantFps);
          if (fpsHistoryRef.current.length > 10) {
            fpsHistoryRef.current.shift();
          }
          const avgFps =
            fpsHistoryRef.current.reduce((sum, value) => sum + value, 0) / fpsHistoryRef.current.length;
          setFps(avgFps.toFixed(1));

          processingTimesRef.current.push(processingTime);
          if (processingTimesRef.current.length > 10) {
            processingTimesRef.current.shift();
          }
          const avgTime =
            processingTimesRef.current.reduce((sum, value) => sum + value, 0) /
            processingTimesRef.current.length;
          setTime(`${avgTime.toFixed(0)}ms`);
        },
        setTransportService: () => undefined,
      }) as Pick<
        IStatsDisplay,
        'updateCameraStatus' | 'updateTransportStatus' | 'updateProcessingStats' | 'setTransportService'
      >,
    []
  );

  const toastManager = useMemo(
    () =>
      ({
        success: toast.success,
        error: toast.error,
        warning: toast.warning,
        info: toast.info,
      }) as IToastDisplay,
    [toast.error, toast.info, toast.success, toast.warning]
  );

  const mapFiltersToValueObjects = useCallback(
    (filters: ActiveFilterState[]): FilterData[] =>
      filters.map((filter) => new FilterData(filter.id, { ...filter.parameters })),
    []
  );

  const mapFiltersToGenericSelections = useCallback(
    (filters: ActiveFilterState[]): GenericFilterSelection[] =>
      filters
        .filter((filter) => filter.id !== 'none')
        .map((filter) => {
          const selection = new GenericFilterSelection({
            filterId: filter.id,
            parameters: Object.entries(filter.parameters).map(
              ([parameterId, value]) =>
                new GenericFilterParameterSelection({
                  parameterId,
                  values: [value],
                })
            ),
          });
          return selection;
        }),
    []
  );

  const updateSource = useCallback((sourceId: string, updater: (source: GridSource) => GridSource) => {
    setSources((current) => current.map((source) => (source.id === sourceId ? updater(source) : source)));
  }, []);

  const sendCameraControlRequest = useCallback(
    (sessionId: string, sourceId: string, filters: ActiveFilterState[], accelerator: 'gpu' | 'cpu') => {
      try {
        webrtcService.sendControlRequest(
          sessionId,
          new ProcessImageRequest({
            sessionId,
            genericFilters: mapFiltersToGenericSelections(filters),
            accelerator: toProtocolAccelerator(accelerator),
            apiVersion: '1.0',
          })
        );
      } catch (error) {
        logger.error('Failed to send live camera filter update', {
          'error.message': error instanceof Error ? error.message : String(error),
          'source.id': sourceId,
          'session.id': sessionId,
        });
      }
    },
    [mapFiltersToGenericSelections]
  );

  const createCameraSession = useCallback(
    async (sourceId: string, stream: MediaStream) => {
      if (cameraSessionSourceIdsRef.current.has(sourceId)) {
        return;
      }
      const source = sourcesRef.current.find((item) => item.id === sourceId);
      if (!source || source.sessionId) {
        return;
      }

      cameraSessionSourceIdsRef.current.add(sourceId);
      try {
        const session = await webrtcService.createSession(sourceId, {
          mode: 'camera-mediatrack',
          localStream: stream,
          useDataChannel: true,
          onRemoteStream: (remoteStream) => {
            updateSource(sourceId, (current) => ({
              ...current,
              remoteStream,
            }));
          },
        });

        updateSource(sourceId, (current) => ({
          ...current,
          sessionId: session.getId(),
          sessionMode: session.getMode(),
        }));
        const currentSource = sourcesRef.current.find((item) => item.id === sourceId) ?? source;
        const currentFilters = currentSource.filters.length
          ? currentSource.filters
          : [{ id: 'none', parameters: {} }];
        sendCameraControlRequest(
          session.getId(),
          sourceId,
          currentFilters,
          currentSource.accelerator
        );
      } catch (error) {
        logger.error('Failed to create camera MediaTrack session', {
          'error.message': error instanceof Error ? error.message : String(error),
          'source.id': sourceId,
        });
        toastManager.warning(
          'Camera transport unavailable',
          'Live local preview is active, but the remote camera MediaTrack session could not be established.'
        );
      } finally {
        cameraSessionSourceIdsRef.current.delete(sourceId);
      }
    },
    [sendCameraControlRequest, toastManager, updateSource]
  );

  const emitSelectionState = useCallback(
    (source: GridSource | null) => {
      if (!source) {
        return;
      }
      setSelectedSource(source.number, source.name);
      setActiveFilters(source.filters.map((f) => ({ id: f.id, parameters: { ...f.parameters } })));
      setResolution(source.resolution);
    },
    [setActiveFilters, setResolution, setSelectedSource]
  );

  const selectSourceById = useCallback(
    (sourceId: string) => {
      setSelectedSourceId(sourceId);
      const source = sourcesRef.current.find((item) => item.id === sourceId) ?? null;
      emitSelectionState(source);
    },
    [emitSelectionState]
  );

  const removeSourceById = useCallback((sourceId: string) => {
    const source = sourcesRef.current.find((item) => item.id === sourceId);
    if (!source) {
      return;
    }
    source.transport?.disconnect();
    if (source.sessionId) {
      void webrtcService.closeSession(source.sessionId).catch((error) => {
        logger.error('Failed to close camera MediaTrack session', {
          'error.message': error instanceof Error ? error.message : String(error),
          'source.id': sourceId,
        });
      });
    }
    setSources((current) => current.filter((item) => item.id !== sourceId));
    setSelectedSourceId((currentSelected) => {
      if (currentSelected !== sourceId) {
        return currentSelected;
      }
      const remaining = sourcesRef.current.filter((item) => item.id !== sourceId);
      const nextSource = remaining[0] ?? null;
      if (nextSource) {
        emitSelectionState(nextSource);
        return nextSource.id;
      }
      return null;
    });
  }, [emitSelectionState]);

  const activeTransportService = useMemo(() => {
    if (selectedSourceId) {
      const selected = sources.find((source) => source.id === selectedSourceId);
      if (selected?.transport) {
        return selected.transport;
      }
    }
    return sources.find((source) => source.transport)?.transport ?? null;
  }, [selectedSourceId, sources]);

  const addSource = useCallback(
    (inputSource: InputSource) => {
      if (sourcesRef.current.length >= MAX_SOURCES) {
        toast?.warning('Maximum Sources', `Cannot add more than ${MAX_SOURCES} sources`);
        return;
      }
      const number = nextNumberRef.current;
      nextNumberRef.current += 1;

      const uniqueId = `${inputSource.id}-${number}`;
      const sourceImagePath =
        inputSource.type === 'video' ? inputSource.previewImagePath || '' : inputSource.imagePath;
      const cameraManager = {
        setProcessing: () => undefined,
        getLastFrameTime: () =>
          cameraFrameTimeRef.current[uniqueId] ?? performance.now(),
      };
      const transport =
        inputSource.type === 'camera'
          ? null
          : new WebRTCFrameTransportService(
            uniqueId,
            statsManager as IStatsDisplay,
            cameraManager as never,
            toastManager as IToastDisplay
          );
      if (transport) {
        transport.connect();
        statsManager.setTransportService(transport);
        transport.onFrameResult((data) => {
          if (!data.imageData?.byteLength || data.width <= 0 || data.height <= 0) {
            return;
          }

          updateSource(uniqueId, (source) => ({
            ...source,
            currentImageSrc: frameResponseToDataUrl(
              data.imageData,
              data.width,
              data.height,
              data.channels || 4
            ),
          }));
        });
        transport.onDetectionResult((frame) => {
          updateSource(uniqueId, (source) => ({
            ...source,
            detections: frame.detections,
            detectionImageWidth: frame.imageWidth,
            detectionImageHeight: frame.imageHeight,
          }));
        });
      }

      const source: GridSource = {
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
        filters: [{ id: 'none', parameters: {} }],
        resolution: 'original',
        accelerator: 'gpu',
        videoId: inputSource.type === 'video' ? inputSource.id : undefined,
        detections: [],
        detectionImageWidth: 0,
        detectionImageHeight: 0,
      };

      setSources((current) => [...current, source]);
      if (sourcesRef.current.length === 0) {
        setSelectedSourceId(uniqueId);
        emitSelectionState(source);
      }

      if (inputSource.type === 'video') {
        const tryStartVideo = () => {
          if (transport.isConnected()) {
            transport.sendStartVideo(inputSource.id, mapFiltersToValueObjects(source.filters), 'gpu');
            return;
          }
          setTimeout(tryStartVideo, 100);
        };
        setTimeout(tryStartVideo, 100);
      }
    },
    [emitSelectionState, mapFiltersToValueObjects, statsManager, toastManager, updateSource]
  );

  useEffect(() => {
    if (!ready) {
      return;
    }
    if (defaultSourceInitializedRef.current) {
      return;
    }
    const input = container.getInputSourceService();
    const defaultSource = input.getDefaultSource();
    if (defaultSource && sourcesRef.current.length === 0) {
      defaultSourceInitializedRef.current = true;
      addSource(defaultSource);
    }
  }, [addSource, container, ready]);

  const openDrawer = useCallback(() => {
    const inputSourceService = container.getInputSourceService();
    setAvailableSources(inputSourceService.getSources());
    setIsDrawerOpen(true);
  }, [container]);

  const refreshDrawerSources = useCallback(() => {
    const inputSourceService = container.getInputSourceService();
    setAvailableSources(inputSourceService.getSources());
  }, [container]);

  const onSelectSourceFromDrawer = useCallback(
    (source: InputSource) => {
      addSource(source);
      setIsDrawerOpen(false);
    },
    [addSource]
  );

  const onSelectImage = useCallback((image: StaticImage) => {
    const imagePath = image.path;
    const sourceNumber = pendingSourceNumberForImageChangeRef.current;
    if (sourceNumber !== null) {
      setSources((current) =>
        current.map((source) =>
          source.number === sourceNumber
            ? {
              ...source,
              imagePath,
              originalImageSrc: imagePath,
              currentImageSrc: imagePath,
            }
            : source
        )
      );
    }
    pendingSourceNumberForImageChangeRef.current = null;
    setIsImageSelectorOpen(false);
  }, []);

  useEffect(() => {
    if (!ready || !selectedSourceId) {
      return;
    }
    const selectedSource = sourcesRef.current.find((source) => source.id === selectedSourceId);
    if (!selectedSource) {
      return;
    }
    const normalizedFilters =
      activeFilters.length > 0 ? activeFilters.map((f) => ({ id: f.id, parameters: { ...f.parameters } })) : [{ id: 'none', parameters: {} }];

    updateSource(selectedSource.id, (source) => ({
      ...source,
      filters: normalizedFilters,
      resolution: selectedResolution,
      accelerator: selectedAccelerator,
    }));

    if (selectedSource.type === 'video') {
      if (!selectedSource.transport || !selectedSource.transport.isConnected()) {
        logger.error('Frame transport not connected for selected source', {
          'source.id': selectedSource.id,
        });
        return;
      }
      const videoId = selectedSource.videoId || selectedSource.name;
      selectedSource.transport.sendStopVideo(videoId);
      setTimeout(() => {
        if (selectedSource.transport?.isConnected()) {
          selectedSource.transport.sendStartVideo(videoId, mapFiltersToValueObjects(normalizedFilters), selectedAccelerator);
        }
      }, 200);
      return;
    }
    if (selectedSource.type === 'camera') {
      if (selectedSource.sessionId && webrtcService.isDataChannelOpen(selectedSource.sessionId)) {
        sendCameraControlRequest(
          selectedSource.sessionId,
          selectedSource.id,
          normalizedFilters,
          selectedAccelerator
        );
      }
      return;
    }

    const applyForStatic = async () => {
      if (!selectedSource.transport) {
        logger.error('Static source is missing a WebRTC transport', {
          'source.id': selectedSource.id,
        });
        return;
      }
      try {
        const originalImg = new Image();
        originalImg.crossOrigin = 'anonymous';
        await new Promise<void>((resolve, reject) => {
          originalImg.onload = () => resolve();
          originalImg.onerror = () => reject(new Error('Failed to load original image'));
          originalImg.src = selectedSource.originalImageSrc;
        });
        const originalWidth = originalImg.naturalWidth || originalImg.width || 512;
        const originalHeight = originalImg.naturalHeight || originalImg.height || 512;
        const factor = selectedResolution === 'half' ? 2 : selectedResolution === 'quarter' ? 4 : 1;
        const targetWidth = Math.floor(originalWidth / factor);
        const targetHeight = Math.floor(originalHeight / factor);

        const rasterized = await rasterizeImageToRgb(
          selectedSource.originalImageSrc,
          targetWidth,
          targetHeight
        );

        if (normalizedFilters.length === 1 && normalizedFilters[0].id === 'none') {
          updateSource(selectedSource.id, (source) => ({
            ...source,
            currentImageSrc: frameResponseToDataUrl(
              rasterized,
              targetWidth,
              targetHeight,
              3
            ),
          }));
          return
        }

        const response = await selectedSource.transport.sendSingleImage(
          rasterized,
          targetWidth,
          targetHeight,
          3,
          mapFiltersToValueObjects(normalizedFilters),
          new AcceleratorConfig(selectedAccelerator),
          new GrayscaleAlgorithm('bt601')
        );

        if (
          response.code === 0 &&
          response.imageData?.byteLength &&
          response.width > 0 &&
          response.height > 0
        ) {
          updateSource(selectedSource.id, (source) => ({
            ...source,
            currentImageSrc: frameResponseToDataUrl(
              response.imageData,
              response.width,
              response.height,
              response.channels || 4
            ),
          }));
          return;
        }

        toastManager.error(
          'Static image processing failed',
          response.message || 'The image processor did not return a processed image.'
        );
      } catch (error) {
        logger.error('Error applying static image filter', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
        toastManager.error(
          'Static image processing failed',
          error instanceof Error ? error.message : 'Unknown processing error'
        );
      }
    };
    void applyForStatic();
  }, [
    activeFilters,
    mapFiltersToValueObjects,
    ready,
    selectedAccelerator,
    selectedResolution,
    selectedSourceId,
    toastManager,
    updateSource,
    sendCameraControlRequest,
  ]);

  useEffect(() => {
    // Only tear down sessions when the browser is actually unloading. React 18
    // StrictMode re-runs effect cleanups during development remounts; running
    // disconnect/closeSession there caused session churn (a new WebRTC session
    // was spawned on the next send, leaving orphaned channels on the backend
    // and triggering "Frame timed out" on the frontend).
    const stopAllSources = () => {
      sourcesRef.current.forEach((source) => {
        source.transport?.disconnect();
        if (source.sessionId) {
          void webrtcService.closeSession(source.sessionId);
        }
      });
    };
    window.addEventListener('beforeunload', stopAllSources);
    return () => {
      window.removeEventListener('beforeunload', stopAllSources);
    };
  }, []);

  return (
    <>
      <VideoGrid
        sources={sources.map((source) => ({
          id: source.id,
          number: source.number,
          name: source.name,
          type: source.type,
          imageSrc: source.currentImageSrc || source.imagePath,
          remoteStream: source.remoteStream,
          filters: source.filters,
          detections: source.detections,
          detectionImageWidth: source.detectionImageWidth,
          detectionImageHeight: source.detectionImageHeight,
        }))}
        selectedSourceId={selectedSourceId}
        onSelectSource={selectSourceById}
        onCloseSource={removeSourceById}
        onChangeImageRequest={(_, sourceNumber) => {
          pendingSourceNumberForImageChangeRef.current = sourceNumber;
          const inputSourceService = container.getInputSourceService();
          void inputSourceService
            .listAvailableImages()
            .then((images) => {
              setAvailableImages(images);
              setIsImageSelectorOpen(true);
            })
            .catch((error) => {
              logger.error('Failed to load image options', {
                'error.message': error instanceof Error ? error.message : String(error),
              });
            });
        }}
        onCameraFrame={(sourceId, payload) => {
          cameraFrameTimeRef.current[sourceId] = payload.timestamp;
          const source = sourcesRef.current.find((item) => item.id === sourceId);
          if (!source || source.sessionMode === 'camera-mediatrack') {
            return;
          }
          if (!source?.transport?.isConnected()) {
            return;
          }
          const filters = source.filters.length ? source.filters : [{ id: 'none', parameters: {} }];
          source.transport.sendFrame(
            `data:image/jpeg;base64,${payload.base64data}`,
            payload.width,
            payload.height,
            mapFiltersToValueObjects(filters),
            selectedAccelerator
          );
        }}
        onCameraStreamReady={createCameraSession}
        onCameraStatus={(status, type) => statsManager.updateCameraStatus(status, type)}
        onCameraError={(title, message) => toastManager.error(title, message)}
        data-testid="video-grid-host"
      />
      <AddSourceFab onClick={openDrawer} />
      <AcceleratorStatusFab />
      <SourceDrawer
        isOpen={isDrawerOpen}
        availableSources={availableSources}
        onClose={() => setIsDrawerOpen(false)}
        onSelectSource={onSelectSourceFromDrawer}
        onSourcesChanged={refreshDrawerSources}
      />
      <ImageSelectorModal
        isOpen={isImageSelectorOpen}
        availableImages={availableImages}
        onClose={() => setIsImageSelectorOpen(false)}
        onSelectImage={onSelectImage}
      />
      <ReactStatsPanel
        fps={fps}
        time={time}
        frames={frames}
        cameraStatus={cameraStatus}
        cameraStatusType={cameraStatusType}
        transportService={activeTransportService}
      />
    </>
  );
}
