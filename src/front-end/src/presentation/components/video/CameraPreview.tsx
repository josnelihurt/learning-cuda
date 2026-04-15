import { useEffect, useRef } from 'react';
import type { ActiveFilterState } from '@/presentation/components/filters/FilterPanel';

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
};

export function CameraPreview({
  width = 640,
  height = 480,
  fps = 15,
  quality = 0.7,
  captureFrames = true,
  remoteStream = null,
  activeFilters = [],
  onFrameCaptured,
  onStreamReady,
  onCameraStatus,
  onCameraError,
}: CameraPreviewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const onFrameCapturedRef = useRef(onFrameCaptured);
  const onStreamReadyRef = useRef(onStreamReady);
  const onCameraStatusRef = useRef(onCameraStatus);
  const onCameraErrorRef = useRef(onCameraError);

  useEffect(() => {
    onFrameCapturedRef.current = onFrameCaptured;
  }, [onFrameCaptured]);

  useEffect(() => {
    onCameraStatusRef.current = onCameraStatus;
  }, [onCameraStatus]);

  useEffect(() => {
    onStreamReadyRef.current = onStreamReady;
  }, [onStreamReady]);

  useEffect(() => {
    onCameraErrorRef.current = onCameraError;
  }, [onCameraError]);

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

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
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

        canvas.width = width;
        canvas.height = height;
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
            ctx.drawImage(video, 0, 0, width, height);
            const dataUrl = canvas.toDataURL('image/jpeg', quality);
            const base64data = dataUrl.split(',')[1] ?? '';
            onFrameCapturedRef.current({ base64data, width, height, timestamp });
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
  }, [captureFrames, fps, height, quality, width]);

  return (
    <>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          background: 'black',
        }}
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </>
  );
}
