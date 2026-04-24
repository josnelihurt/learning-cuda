import { useMemo, useRef, useState } from 'react';
import type { IStatsDisplay } from '@/infrastructure/transport/transport-types';

type StatsManager = Pick<IStatsDisplay, 'updateCameraStatus' | 'updateTransportStatus' | 'updateProcessingStats'>;

type ProcessingStatsResult = {
  fps: string;
  time: string;
  frames: number;
  cameraStatus: string;
  cameraStatusType: 'success' | 'error' | 'warning' | 'inactive';
  statsManager: StatsManager;
};

export function useProcessingStats(): ProcessingStatsResult {
  const [fps, setFps] = useState('--');
  const [time, setTime] = useState('--ms');
  const [frames, setFrames] = useState(0);
  const [cameraStatus, setCameraStatus] = useState('Inactive');
  const [cameraStatusType, setCameraStatusType] = useState<'success' | 'error' | 'warning' | 'inactive'>('inactive');
  const fpsHistoryRef = useRef<number[]>([]);
  const processingTimesRef = useRef<number[]>([]);

  const statsManager = useMemo<StatsManager>(() => ({
    updateCameraStatus: (status: string, type: 'success' | 'error' | 'warning' | 'inactive'): void => {
      setCameraStatus(status);
      setCameraStatusType(type);
    },
    updateTransportStatus: (): void => undefined,
    updateProcessingStats: (processingTime: number): void => {
      setFrames((current) => current + 1);
      const instantFps = 1000 / processingTime;
      fpsHistoryRef.current.push(instantFps);
      if (fpsHistoryRef.current.length > 10) fpsHistoryRef.current.shift();
      const avgFps = fpsHistoryRef.current.reduce((sum, v) => sum + v, 0) / fpsHistoryRef.current.length;
      setFps(avgFps.toFixed(1));
      processingTimesRef.current.push(processingTime);
      if (processingTimesRef.current.length > 10) processingTimesRef.current.shift();
      const avgTime = processingTimesRef.current.reduce((sum, v) => sum + v, 0) / processingTimesRef.current.length;
      setTime(`${avgTime.toFixed(0)}ms`);
    },
  }), []);

  return { fps, time, frames, cameraStatus, cameraStatusType, statsManager };
}
