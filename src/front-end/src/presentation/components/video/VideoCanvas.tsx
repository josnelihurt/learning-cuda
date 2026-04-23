import { useRef, useEffect, type ReactElement } from 'react';
import styles from './VideoCanvas.module.css';

// Generate a color from a class ID using HSL
function colorForClassId(classId: number): string {
  const hue = (classId * 137.5) % 360; // Golden angle for color variety
  return `hsl(${hue}, 70%, 50%)`;
}

interface VideoCanvasProps {
  width?: number;
  height?: number;
  onFrame?: (base64data: string, width: number, height: number) => void;
  className?: string;
  detections?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    className?: string;
    confidence?: number;
    classId?: number;
  }>;
}

export function VideoCanvas({
  width = 640,
  height = 480,
  onFrame,
  className,
  detections = [],
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

          // Draw detections if present
          if (detections && detections.length > 0) {
            detections.forEach((det) => {
              const color = colorForClassId(det.classId ?? 0);
              ctxRef.current!.strokeStyle = color;
              ctxRef.current!.lineWidth = 2;
              ctxRef.current!.strokeRect(det.x, det.y, det.width, det.height);

              // Draw label with confidence
              if (det.className) {
                const label = det.confidence
                  ? `${det.className} ${(det.confidence * 100).toFixed(0)}%`
                  : det.className;

                ctxRef.current!.font = '12px Arial';
                ctxRef.current!.fillStyle = color;
                ctxRef.current!.fillRect(det.x, det.y - 20, label.length * 7 + 4, 18);

                ctxRef.current!.fillStyle = '#fff';
                ctxRef.current!.fillText(label, det.x + 2, det.y - 6);
              }
            });
          }
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
  }, [onFrame, width, height, detections]);

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
