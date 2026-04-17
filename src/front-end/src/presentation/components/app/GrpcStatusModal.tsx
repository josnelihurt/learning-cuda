import { useEffect, useState, type ReactElement } from 'react';
import { remoteManagementService } from '@/infrastructure/external/remote-management-service';
import { AcceleratorHealthStatus } from '@/gen/remote_management_service_pb';
import { logger } from '@/infrastructure/observability/otel-logger';
import styles from './GrpcStatusModal.module.css';

declare global {
  interface Window {
    __grpcStatusModal?: {
      isOpen: () => boolean;
      isMinimized: () => boolean;
      open: () => void;
      restore: () => void;
    };
  }
}

export function GrpcStatusModal(): ReactElement {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [terminalOutput, setTerminalOutput] = useState<string[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const [isHealthy, setIsHealthy] = useState(false);

  useEffect(() => {
    const api = {
      isOpen: () => isOpen,
      isMinimized: () => isMinimized,
      open: () => {
        setIsMinimized(false);
        setIsOpen(true);
      },
      restore: () => {
        setIsMinimized(false);
        setIsOpen(true);
      },
    };
    window.__grpcStatusModal = api;
    return () => {
      if (window.__grpcStatusModal === api) {
        delete window.__grpcStatusModal;
      }
    };
  }, [isMinimized, isOpen]);

  useEffect(() => {
    const openModal = () => {
      setIsMinimized(false);
      setIsOpen(true);
    };
    document.addEventListener('open-grpc-status-modal', openModal);
    document.addEventListener('accelerator-unhealthy', openModal);
    return () => {
      document.removeEventListener('open-grpc-status-modal', openModal);
      document.removeEventListener('accelerator-unhealthy', openModal);
    };
  }, []);

  useEffect(() => {
    const check = async () => {
      try {
        const health = await remoteManagementService.checkAcceleratorHealth();
        setIsHealthy(health.status === AcceleratorHealthStatus.HEALTHY);
      } catch (error) {
        logger.error('Failed to check accelerator health in React modal', {
          'error.message': error instanceof Error ? error.message : String(error),
        });
        setIsHealthy(false);
      }
    };
    void check();
    const interval = window.setInterval(() => {
      void check();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!isOpen) {
    return null as unknown as ReactElement;
  }

  return (
    <div className={styles.overlay}>
      <div
        className={styles.backdrop}
        onClick={() => {
          setIsOpen(false);
          setIsMinimized(true);
        }}
      />
      <div className={styles.panel}>
        <div className={styles.header}>
          <h2 className={styles.title}>
            <span className={isHealthy ? styles.healthDotOk : styles.healthDotBad} />
            CUDA Accelerator Micro Service Status
          </h2>
          <button type="button" className="feature-flags-btn" onClick={() => { setIsOpen(false); setIsMinimized(true); }}>
            ×
          </button>
        </div>
        <div className={styles.body}>
          <p>
            This project is hosted on a cloud VM without GPU. The NVIDIA GPU is located at home and can be powered on remotely from this panel.
          </p>
          <div className={styles.terminal}>
            {terminalOutput.length === 0 ? 'Terminal output will appear here when starting the Jetson Nano...' : terminalOutput.map((line) => <div key={line}>{line}</div>)}
          </div>
          <div className={styles.actions}>
            <button
              type="button"
              className="feature-flags-btn"
              disabled={isStarting}
              onClick={() => {
                if (isStarting) {
                  return;
                }
                setIsStarting(true);
                setTerminalOutput((current) => [...current, `[${new Date().toLocaleTimeString()}] Sending POWER ON command via MQTT...`]);
                void remoteManagementService
                  .startJetsonNano(
                    (event) => {
                      const message = event.message || event.status || 'No message';
                      setTerminalOutput((current) => [...current, `[${new Date().toLocaleTimeString()}] ${message}`].slice(-100));
                    },
                    (error) => {
                      setTerminalOutput((current) => [...current, `[${new Date().toLocaleTimeString()}] Error: ${error.message}`].slice(-100));
                    }
                  )
                  .catch((error) => {
                    setTerminalOutput((current) => [...current, `[${new Date().toLocaleTimeString()}] Error: ${error instanceof Error ? error.message : String(error)}`].slice(-100));
                  })
                  .finally(() => {
                    setIsStarting(false);
                  });
              }}
            >
              {isStarting ? 'Powering on...' : 'Power On'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
