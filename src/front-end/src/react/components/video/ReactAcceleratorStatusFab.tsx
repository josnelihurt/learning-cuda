import React from 'react';
import { useEffect, useState } from 'react';
import { remoteManagementService } from '@/infrastructure/external/remote-management-service';
import { AcceleratorHealthStatus } from '@/gen/remote_management_service_pb';
import { logger } from '@/infrastructure/observability/otel-logger';

declare global {
  interface Window {
    __reactGrpcStatusModal?: {
      isOpen: () => boolean;
      isMinimized: () => boolean;
      open: () => void;
      restore: () => void;
    };
  }
}

export function ReactAcceleratorStatusFab() {
  const [isVisible, setIsVisible] = useState(false);
  const [isBlinking, setIsBlinking] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const checkHealth = async () => {
      const isModalOpen = Boolean(window.__reactGrpcStatusModal?.isOpen());
      try {
        const response = await remoteManagementService.checkAcceleratorHealth();
        const healthy = response.status === AcceleratorHealthStatus.HEALTHY;
        if (cancelled) {
          return;
        }
        setIsBlinking(!healthy);
        setIsVisible(!healthy && !isModalOpen);
      } catch (error) {
        logger.error('Failed to check accelerator health in React FAB', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
        if (!cancelled) {
          setIsBlinking(true);
          setIsVisible(!isModalOpen);
        }
      }
    };

    void checkHealth();
    const interval = window.setInterval(() => {
      void checkHealth();
    }, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (!isVisible) {
    return null;
  }

  return (
    <button
      type="button"
      className={`react-accelerator-fab ${isBlinking ? 'blinking' : ''}`}
      data-testid="accelerator-status-fab"
      onClick={() => {
        const modalElement = window.__reactGrpcStatusModal;
        if (!modalElement) {
          return;
        }
        if (modalElement.isMinimized()) {
          modalElement.restore();
        } else {
          modalElement.open();
        }
      }}
    >
      <span className="fab-icon">!</span>
      <span className="fab-text">Accelerator Offline</span>
    </button>
  );
}
