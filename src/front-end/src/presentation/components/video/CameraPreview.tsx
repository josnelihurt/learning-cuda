import { useEffect, useRef, type ReactElement } from 'react';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';
import { useLatest } from '@/presentation/hooks/useLatest';
import styles from './CameraPreview.module.css';

type FrameCapturedPayload = {
  base64data: string;
  width: number;
  height: number;
  timestamp: number;
};

type CameraPreviewProps = {
  width?: number;
  height?: number;
  fps?: number;
  quality?: number;
  captureFrames?: boolean;
  remoteStream?: MediaStream | null;
  activeFilters?: ActiveFilterState[];
  onFrameCaptured: (payload: FrameCapturedPayload) => void;
  onStreamReady?: (stream: MediaStream) => void;
  onCameraStatus: (status: string, type: 'success' | 'error' | 'warning' | 'inactive') => void;
  onCameraError: (title: string, message: string) => void;
  onCameraPermissionDenied?: () => void;
  onFpsUpdate?: (fps: number) => void;
  onResolutionUpdate?: (width: number, height: number) => void;
};

export function CameraPreview({
  width,
  height,
  fps = 15,
  quality = 0.7,
  captureFrames = true,
  remoteStream = null,
  activeFilters = [],
  onFrameCaptured,
  onStreamReady,
  onCameraStatus,
  onCameraError,
  onCameraPermissionDenied,
  onFpsUpdate,
  onResolutionUpdate,
}: CameraPreviewProps): ReactElement {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const onFrameCapturedRef = useLatest(onFrameCaptured);
  const onStreamReadyRef = useLatest(onStreamReady);
  const onCameraStatusRef = useLatest(onCameraStatus);
  const onCameraErrorRef = useLatest(onCameraError);
  const onFpsUpdateRef = useLatest(onFpsUpdate);
  const onResolutionUpdateRef = useLatest(onResolutionUpdate);
  const displayFrameTimesRef = useRef<number[]>([]);
  const lastFpsEmitAtRef = useRef(0);
  const initialWidthRef = useRef<number | undefined>(width);
  const initialHeightRef = useRef<number | undefined>(height);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    if (!remoteStream) {
      video.srcObject = streamRef.current;
      if (video.srcObject) {
        void video.play().catch(() => undefined);
      }
      return;
    }

    let cancelled = false;
    const probe = document.createElement('video');
    probe.muted = true;
    probe.playsInline = true;
    probe.srcObject = remoteStream;

    const tryPromoteRemoteStream = async () => {
      try {
        await probe.play();
      } catch {
        return;
      }

      const isPlayable = await new Promise<boolean>((resolve) => {
        const deadline = window.setTimeout(() => resolve(false), 1500);

        const checkPlayback = () => {
          if (probe.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA || probe.videoWidth > 0) {
            clearTimeout(deadline);
            resolve(true);
            return;
          }
          window.requestAnimationFrame(checkPlayback);
        };

        checkPlayback();
      });

      if (!cancelled && isPlayable) {
        video.srcObject = remoteStream;
        void video.play().catch(() => undefined);
        const measuredWidth = probe.videoWidth || video.videoWidth;
        const measuredHeight = probe.videoHeight || video.videoHeight;
        if (measuredWidth > 0 && measuredHeight > 0) {
          onResolutionUpdateRef.current?.(measuredWidth, measuredHeight);
        } else {
          const trackSettings = remoteStream.getVideoTracks()[0]?.getSettings();
          if ((trackSettings?.width ?? 0) > 0 && (trackSettings?.height ?? 0) > 0) {
            onResolutionUpdateRef.current?.(trackSettings!.width!, trackSettings!.height!);
          }
        }
      }
    };

    void tryPromoteRemoteStream();

    return () => {
      cancelled = true;
      probe.srcObject = null;
    };
  }, [remoteStream]);

  useEffect(() => {
    let cancelled = false;

    const startCamera = async () => {
      try {
        onCameraStatusRef.current('Requesting access...', 'warning');
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Camera API not available. Use HTTPS.');
        }

        const initialWidth = initialWidthRef.current;
        const initialHeight = initialHeightRef.current;
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            ...(typeof initialWidth === 'number' ? { width: { ideal: initialWidth } } : {}),
            ...(typeof initialHeight === 'number' ? { height: { ideal: initialHeight } } : {}),
          },
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) {
          throw new Error('Camera preview elements are not ready');
        }

        video.srcObject = stream;
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve();
        });
        await video.play();
        onStreamReadyRef.current?.(stream);
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          onResolutionUpdateRef.current?.(video.videoWidth, video.videoHeight);
        } else {
          const trackSettings = stream.getVideoTracks()[0]?.getSettings();
          onResolutionUpdateRef.current?.(
            trackSettings?.width ?? initialWidth ?? 0,
            trackSettings?.height ?? initialHeight ?? 0
          );
        }

        const captureWidth = initialWidth ?? video.videoWidth;
        const captureHeight = initialHeight ?? video.videoHeight;
        canvas.width = captureWidth;
        canvas.height = captureHeight;
        onCameraStatusRef.current('Active', 'success');

        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) {
          throw new Error('Camera canvas context is not available');
        }

        if (captureFrames) {
          intervalRef.current = window.setInterval(() => {
            if (!video.videoWidth) {
              return;
            }
            const timestamp = performance.now();
            const frameWidth = initialWidth ?? video.videoWidth;
            const frameHeight = initialHeight ?? video.videoHeight;
            ctx.drawImage(video, 0, 0, frameWidth, frameHeight);
            const dataUrl = canvas.toDataURL('image/jpeg', quality);
            const base64data = dataUrl.split(',')[1] ?? '';
            onFrameCapturedRef.current({ base64data, width: frameWidth, height: frameHeight, timestamp });
          }, 1000 / fps);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        onCameraErrorRef.current('Camera Error', message);
        onCameraStatusRef.current('Camera Error', 'error');
      }
    };

    void startCamera();
    return () => {
      cancelled = true;
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      onCameraStatusRef.current('Inactive', 'inactive');
    };
  }, [captureFrames, fps, quality]);

  useEffect(() => {
    const stream = streamRef.current;
    if (!stream) {
      return;
    }
    const videoTrack = stream.getVideoTracks()[0];
    if (!videoTrack) {
      return;
    }
    const constraints: MediaTrackConstraints = {
      ...(typeof width === 'number' ? { width: { ideal: width } } : {}),
      ...(typeof height === 'number' ? { height: { ideal: height } } : {}),
    };
    if (Object.keys(constraints).length === 0) {
      return;
    }
    videoTrack
      .applyConstraints(constraints)
      .then(() => {
        const settings = videoTrack.getSettings();
        if ((settings.width ?? 0) > 0 && (settings.height ?? 0) > 0) {
          onResolutionUpdateRef.current?.(settings.width!, settings.height!);
        }
      })
      .catch(() => {
        // applyConstraints can fail if the camera doesn't support the exact size — keep current resolution.
      });
  }, [width, height, onResolutionUpdateRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    let cancelled = false;
    let rafId: number | null = null;

    const emitFps = (now: number): void => {
      const frameTimes = displayFrameTimesRef.current;
      frameTimes.push(now);
      const cutoff = now - 1000;
      while (frameTimes.length > 0 && frameTimes[0] < cutoff) {
        frameTimes.shift();
      }

      const lastEmitAt = lastFpsEmitAtRef.current;
      if (now - lastEmitAt < 200 || frameTimes.length < 2) {
        return;
      }
      const durationMs = Math.max(1, frameTimes[frameTimes.length - 1] - frameTimes[0]);
      const currentFps = ((frameTimes.length - 1) * 1000) / durationMs;
      lastFpsEmitAtRef.current = now;
      onFpsUpdateRef.current?.(currentFps);
    };

    const scheduleRafLoop = (): void => {
      let lastVideoTime = -1;
      const tick = (): void => {
        if (cancelled) {
          return;
        }
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;
          emitFps(performance.now());
        }
        rafId = window.requestAnimationFrame(tick);
      };
      rafId = window.requestAnimationFrame(tick);
    };

    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      const withVideoFrames = video as HTMLVideoElement & {
        requestVideoFrameCallback: (callback: (now: DOMHighResTimeStamp) => void) => number;
      };
      const onVideoFrame = (now: DOMHighResTimeStamp): void => {
        if (cancelled) {
          return;
        }
        emitFps(now);
        withVideoFrames.requestVideoFrameCallback(onVideoFrame);
      };
      withVideoFrames.requestVideoFrameCallback(onVideoFrame);
    } else {
      scheduleRafLoop();
    }

    return () => {
      cancelled = true;
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
      }
      displayFrameTimesRef.current = [];
      lastFpsEmitAtRef.current = 0;
    };
  }, []);

  return (
    <>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={styles.preview}
      />
      <canvas ref={canvasRef} className={styles.hiddenCanvas} />
    </>
  );
}
