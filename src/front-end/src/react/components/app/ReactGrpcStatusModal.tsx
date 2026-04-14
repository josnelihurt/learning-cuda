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

export function ReactGrpcStatusModal() {
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
    window.__reactGrpcStatusModal = api;
    return () => {
      if (window.__reactGrpcStatusModal === api) {
        delete window.__reactGrpcStatusModal;
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
    return null;
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 10000 }}>
      <div
        style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.6)' }}
        onClick={() => {
          setIsOpen(false);
          setIsMinimized(true);
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '90vw',
          height: '85vh',
          maxWidth: '1198px',
          maxHeight: '800px',
          background: 'rgba(20, 20, 28, 0.95)',
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column',
          color: '#fff',
        }}
      >
        <div style={{ padding: '12px 20px', display: 'flex', justifyContent: 'space-between' }}>
          <h2 style={{ margin: 0, fontSize: '16px' }}>
            <span
              style={{
                display: 'inline-block',
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                marginRight: '8px',
                background: isHealthy ? '#22c55e' : '#ef4444',
              }}
            />
            CUDA Accelerator Micro Service Status
          </h2>
          <button type="button" className="feature-flags-btn" onClick={() => { setIsOpen(false); setIsMinimized(true); }}>
            ×
          </button>
        </div>
        <div style={{ padding: '20px', overflowY: 'auto', flex: 1 }}>
          <p>
            This project is hosted on a cloud VM without GPU. The NVIDIA GPU is located at home and can be powered on remotely from this panel.
          </p>
          <div style={{ background: 'rgba(0, 0, 0, 0.5)', borderRadius: '8px', padding: '12px', minHeight: '240px' }}>
            {terminalOutput.length === 0 ? 'Terminal output will appear here when starting the Jetson Nano...' : terminalOutput.map((line) => <div key={line}>{line}</div>)}
          </div>
          <div style={{ marginTop: '16px' }}>
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
