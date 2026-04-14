import { useEffect, useRef } from 'react';

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
  onFrameCaptured: (payload: FrameCapturedPayload) => void;
  onCameraStatus: (status: string, type: 'success' | 'error' | 'warning' | 'inactive') => void;
  onCameraError: (title: string, message: string) => void;
};

export function CameraPreview({
  width = 640,
  height = 480,
  fps = 15,
  quality = 0.7,
  onFrameCaptured,
  onCameraStatus,
  onCameraError,
}: CameraPreviewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const onFrameCapturedRef = useRef(onFrameCaptured);
  const onCameraStatusRef = useRef(onCameraStatus);
  const onCameraErrorRef = useRef(onCameraError);

  useEffect(() => {
    onFrameCapturedRef.current = onFrameCaptured;
  }, [onFrameCaptured]);

  useEffect(() => {
    onCameraStatusRef.current = onCameraStatus;
  }, [onCameraStatus]);

  useEffect(() => {
    onCameraErrorRef.current = onCameraError;
  }, [onCameraError]);

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

        canvas.width = width;
        canvas.height = height;
        onCameraStatusRef.current('Active', 'success');

        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        if (!ctx) {
          throw new Error('Camera canvas context is not available');
        }

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
  }, [fps, height, quality, width]);

  return (
    <>
      <video ref={videoRef} autoPlay playsInline muted style={{ position: 'absolute', opacity: 0 }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </>
  );
}
