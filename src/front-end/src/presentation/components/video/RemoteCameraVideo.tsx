import { useCallback, useEffect, useRef, type ReactElement } from 'react';
import styles from './CameraPreview.module.css';

type RemoteCameraVideoProps = {
  stream: MediaStream;
  onResolutionUpdate?: (width: number, height: number) => void;
  onFpsUpdate?: (fps: number) => void;
};

export function RemoteCameraVideo({ stream, onResolutionUpdate, onFpsUpdate }: RemoteCameraVideoProps): ReactElement {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastReportedResolutionRef = useRef<{ width: number; height: number } | null>(null);
  const fpsFrameCountRef = useRef(0);
  const fpsLastTimeRef = useRef(0);
  const rafIdRef = useRef(0);

  const reportResolution = useCallback((video: HTMLVideoElement) => {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w <= 0 || h <= 0) return;
    const last = lastReportedResolutionRef.current;
    if (last && last.width === w && last.height === h) return;
    lastReportedResolutionRef.current = { width: w, height: h };
    onResolutionUpdate?.(w, h);
  }, [onResolutionUpdate]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.srcObject !== stream) {
      video.srcObject = stream;
      void video.play().catch(() => undefined);
    }
  }, [stream]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onLoadedMetadata = () => {
      reportResolution(video);
    };

    const onResize = () => {
      reportResolution(video);
    };

    video.addEventListener('loadedmetadata', onLoadedMetadata);
    video.addEventListener('resize', onResize);
    reportResolution(video);

    return () => {
      video.removeEventListener('loadedmetadata', onLoadedMetadata);
      video.removeEventListener('resize', onResize);
    };
  }, [reportResolution]);

  useEffect(() => {
    if (!onFpsUpdate) return;

    const video = videoRef.current;
    if (!video) return;

    const tick = (now: number, metadata: VideoFrameCallbackMetadata) => {
      void metadata;
      fpsFrameCountRef.current++;
      if (fpsLastTimeRef.current === 0) {
        fpsLastTimeRef.current = now;
      }
      const elapsed = now - fpsLastTimeRef.current;
      if (elapsed >= 1000) {
        const fps = (fpsFrameCountRef.current * 1000) / elapsed;
        onFpsUpdate(fps);
        fpsFrameCountRef.current = 0;
        fpsLastTimeRef.current = now;
      }
      rafIdRef.current = video.requestVideoFrameCallback(tick);
    };

    rafIdRef.current = video.requestVideoFrameCallback(tick);

    return () => {
      video.cancelVideoFrameCallback(rafIdRef.current);
    };
  }, [onFpsUpdate]);

  useEffect(() => {
    const video = videoRef.current;
    return () => {
      if (video) {
        video.srcObject = null;
      }
    };
  }, []);

  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted
      className={styles.preview}
      data-testid="remote-camera-video"
    />
  );
}
