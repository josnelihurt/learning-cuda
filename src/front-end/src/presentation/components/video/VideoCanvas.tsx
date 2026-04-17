import { useRef, useEffect, type ReactElement } from 'react';
import styles from './VideoCanvas.module.css';

interface VideoCanvasProps {
  width?: number;
  height?: number;
  onFrame?: (base64data: string, width: number, height: number) => void;
  className?: string;
}

export function VideoCanvas({
  width = 640,
  height = 480,
  onFrame,
  className,
}: VideoCanvasProps): ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const fpsRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(0);

  // Initialize canvas context once
  useEffect(() => {
    if (canvasRef.current) {
      ctxRef.current = canvasRef.current.getContext('2d', {
        willReadFrequently: true,
      });
    }
  }, []);

  // Frame rendering via requestAnimationFrame per D-06
  useEffect(() => {
    if (!onFrame || !ctxRef.current || !canvasRef.current) {
      return;
    }

    let animationFrameId: number;

    const render = (timestamp: number) => {
      // Calculate FPS per D-07
      frameCountRef.current++;
      if (timestamp - lastFpsUpdateRef.current >= 1000) {
        fpsRef.current = Math.round(
          (frameCountRef.current * 1000) / (timestamp - lastFpsUpdateRef.current)
        );
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = timestamp;
      }

      // Request next frame
      animationFrameId = requestAnimationFrame(render);
    };

    // Start rendering loop
    animationFrameId = requestAnimationFrame(render);

    // Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [onFrame]);

  // Handle incoming frames
  useEffect(() => {
    if (!onFrame || !ctxRef.current || !canvasRef.current) {
      // Clear canvas if no frame callback
      if (ctxRef.current && canvasRef.current) {
        ctxRef.current.clearRect(0, 0, width, height);
      }
      return;
    }

    // Set up frame handler
    const handleFrame = (base64data: string, w: number, h: number) => {
      if (!ctxRef.current || !canvasRef.current) return;

      const img = new Image();
      img.onload = () => {
        if (ctxRef.current) {
          ctxRef.current.drawImage(img, 0, 0, w, h);
        }
      };
      img.src = `data:image/jpeg;base64,${base64data}`;
    };

    // Call onFrame to register frame handler
    // Note: This is a simplified pattern - in practice, the hook/component
    // will call this handler when frames arrive via WebRTC data channel
    onFrame('', width, height); // Register handler

    // Cleanup
    return () => {
      // Clear canvas
      if (ctxRef.current && canvasRef.current) {
        ctxRef.current.clearRect(0, 0, width, height);
      }
    };
  }, [onFrame, width, height]);

  return (
    <div className={`${styles.canvasContainer} ${className || ''}`} data-testid="video-canvas-container">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className={styles.canvas}
        aria-label="Video stream"
        data-testid="video-canvas"
      />
      {fpsRef.current > 0 && (
        <div className={styles.fpsCounter} data-testid="fps-counter">
          {fpsRef.current} FPS
        </div>
      )}
    </div>
  );
}
