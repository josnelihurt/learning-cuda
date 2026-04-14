import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { InputSource } from '@/gen/config_service_pb';
import type { StaticImage } from '@/gen/common_pb';
import type { StatsPanel } from '@/lit/components/app/stats-panel';
import type { ToastContainer } from '@/lit/components/app/toast-container';
import type { ActiveFilterState } from '../filters/FilterPanel';
import { useAppServices } from '../../providers/app-services-provider';
import { useDashboardState } from '../../context/dashboard-state-context';
import { WebSocketService } from '@/infrastructure/transport/websocket-frame-transport';
import { FilterData } from '@/domain/value-objects';
import { webrtcService } from '@/infrastructure/connection/webrtc-service';
import { logger } from '@/infrastructure/observability/otel-logger';
import { VideoGrid } from './VideoGrid';
import { SourceDrawer } from './SourceDrawer';
import { AddSourceFab } from './AddSourceFab';
import { ImageSelectorModal } from './ImageSelectorModal';
import { AcceleratorStatusFab } from './AcceleratorStatusFab';
import { useToast } from '../../hooks/useToast';
import { StatsPanel } from '../app/StatsPanel';

type GridSource = {
  id: string;
  number: number;
  name: string;
  type: string;
  imagePath: string;
  originalImageSrc: string;
  currentImageSrc: string;
  ws: WebSocketService | null;
  filters: ActiveFilterState[];
  resolution: string;
  videoId?: string;
  webrtcSessionId?: string;
};

const MAX_SOURCES = 9;

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
        updateWebSocketStatus: () => undefined,
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
        setWebSocketService: () => undefined,
      }) as Pick<
        StatsPanel,
        'updateCameraStatus' | 'updateWebSocketStatus' | 'updateProcessingStats' | 'setWebSocketService'
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
      }) as ToastContainer,
    [toast.error, toast.info, toast.success, toast.warning]
  );

  const mapFiltersToValueObjects = useCallback(
    (filters: ActiveFilterState[]): FilterData[] =>
      filters.map((filter) => new FilterData(filter.id, { ...filter.parameters })),
    []
  );

  const updateSource = useCallback((sourceId: string, updater: (source: GridSource) => GridSource) => {
    setSources((current) => current.map((source) => (source.id === sourceId ? updater(source) : source)));
  }, []);

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
    if (source.webrtcSessionId) {
      webrtcService.stopHeartbeat(source.webrtcSessionId);
      void webrtcService.closeSession(source.webrtcSessionId).catch((error) => {
        logger.error('Failed to close WebRTC session', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      });
    }
    source.ws?.disconnect();
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
      const ws = new WebSocketService(statsManager as StatsPanel, cameraManager as never, toastManager);
      ws?.connect();
      statsManager.setWebSocketService(ws);
      if (ws) {
        ws.onFrameResult((data) => {
          const frameData = data.videoFrame ?? data.response;
          if (!frameData) {
            return;
          }
          const bytes = (frameData as { imageData?: Uint8Array; frameData?: Uint8Array }).imageData ??
            (frameData as { imageData?: Uint8Array; frameData?: Uint8Array }).frameData;
          if (!bytes) {
            return;
          }
          let binary = '';
          for (let index = 0; index < bytes.byteLength; index += 1) {
            binary += String.fromCharCode(bytes[index]);
          }
          updateSource(uniqueId, (source) => ({
            ...source,
            currentImageSrc: `data:image/png;base64,${btoa(binary)}`,
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
        ws,
        filters: [{ id: 'none', parameters: {} }],
        resolution: 'original',
        videoId: inputSource.type === 'video' ? inputSource.id : undefined,
      };

      setSources((current) => [...current, source]);
      if (sourcesRef.current.length === 0) {
        setSelectedSourceId(uniqueId);
        emitSelectionState(source);
      }

      if (inputSource.type === 'video') {
        const tryStartVideo = () => {
          if (ws?.isConnected()) {
            ws.sendStartVideo(inputSource.id, mapFiltersToValueObjects(source.filters), 'gpu');
            return;
          }
          setTimeout(tryStartVideo, 100);
        };
        setTimeout(tryStartVideo, 100);
      }

      void webrtcService.createSession(uniqueId).then((session) => {
        updateSource(uniqueId, (current) => ({ ...current, webrtcSessionId: session.getId() }));
      }).catch((error) => {
        logger.error('Failed to create WebRTC session', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
      });
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
    }));

    if (!selectedSource.ws || !selectedSource.ws.isConnected()) {
      logger.error('WebSocket not connected for selected source', {
        'source.id': selectedSource.id,
      });
      return;
    }
    if (selectedSource.type === 'video') {
      const videoId = selectedSource.videoId || selectedSource.name;
      selectedSource.ws.sendStopVideo(videoId);
      setTimeout(() => {
        if (selectedSource.ws?.isConnected()) {
          selectedSource.ws.sendStartVideo(videoId, mapFiltersToValueObjects(normalizedFilters), selectedAccelerator);
        }
      }, 200);
      return;
    }
    if (selectedSource.type === 'camera') {
      return;
    }

    const applyForStatic = async () => {
      try {
        const originalImg = new Image();
        originalImg.crossOrigin = 'anonymous';
        await new Promise<void>((resolve, reject) => {
          originalImg.onload = () => resolve();
          originalImg.onerror = () => reject(new Error('Failed to load original image'));
          originalImg.src = selectedSource.originalImageSrc;
        });
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          return;
        }
        const originalWidth = originalImg.naturalWidth || originalImg.width || 512;
        const originalHeight = originalImg.naturalHeight || originalImg.height || 512;
        const factor = selectedResolution === 'half' ? 2 : selectedResolution === 'quarter' ? 4 : 1;
        const targetWidth = Math.floor(originalWidth / factor);
        const targetHeight = Math.floor(originalHeight / factor);
        canvas.width = targetWidth;
        canvas.height = targetHeight;
        ctx.drawImage(originalImg, 0, 0, targetWidth, targetHeight);
        const imageData = canvas.toDataURL('image/png');
        const response = await selectedSource.ws!.sendSingleFrame(
          imageData,
          targetWidth,
          targetHeight,
          mapFiltersToValueObjects(normalizedFilters),
          selectedAccelerator
        );
        if (response.success && response.response) {
          let binary = '';
          const bytes = response.response.imageData;
          for (let index = 0; index < bytes.byteLength; index += 1) {
            binary += String.fromCharCode(bytes[index]);
          }
          updateSource(selectedSource.id, (source) => ({
            ...source,
            currentImageSrc: `data:image/png;base64,${btoa(binary)}`,
          }));
        }
      } catch (error) {
        logger.error('Error applying static image filter', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
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
    updateSource,
  ]);

  useEffect(() => {
    const stopAllSources = () => {
      sourcesRef.current.forEach((source) => {
        source.ws?.disconnect();
        if (source.webrtcSessionId) {
          webrtcService.stopHeartbeat(source.webrtcSessionId);
          void webrtcService.closeSession(source.webrtcSessionId).catch((error) => {
            logger.error('Failed to close WebRTC session on cleanup', {
              'error.message': error instanceof Error ? error.message : String(error),
            });
          });
        }
      });
    };
    window.addEventListener('beforeunload', stopAllSources);
    return () => {
      window.removeEventListener('beforeunload', stopAllSources);
      stopAllSources();
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
          if (!source?.ws?.isConnected()) {
            return;
          }
          const filters = source.filters.length ? source.filters : [{ id: 'none', parameters: {} }];
          source.ws.sendFrame(
            `data:image/jpeg;base64,${payload.base64data}`,
            payload.width,
            payload.height,
            mapFiltersToValueObjects(filters),
            selectedAccelerator
          );
        }}
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
      <StatsPanel
        fps={fps}
        time={time}
        frames={frames}
        cameraStatus={cameraStatus}
        cameraStatusType={cameraStatusType}
      />
    </>
  );
}
