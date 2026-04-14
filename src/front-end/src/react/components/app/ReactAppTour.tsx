import { useEffect, useMemo, useState } from 'react';
import { logger } from '@/infrastructure/observability/otel-logger';

declare const __APP_VERSION__: string;

const DISMISS_KEY = 'cuda-app-tour-dismissed';
const COMMIT_HASH_KEY = 'cuda-app-tour-commit-hash';

type TourStep = {
  id: string;
  selector: string;
  title: string;
  description: string;
};

const TOUR_STEPS: TourStep[] = [
  {
    id: 'add-source',
    selector: 'button[data-testid="add-input-fab"]',
    title: 'Add Input',
    description: 'Use Add Input to bring images, live camera, or videos into the workspace.',
  },
  {
    id: 'filter-panel',
    selector: '.sidebar .filters-section',
    title: 'Filter Panel',
    description: 'Drag and tune filters. GPU runs on accelerator, CPU runs on server CPU.',
  },
  {
    id: 'tools-dropdown',
    selector: '[data-testid="tools-dropdown-button"]',
    title: 'Tools Menu',
    description: 'Open Tools to access Grafana, Jaeger and operational links.',
  },
  {
    id: 'feature-flags',
    selector: '[data-testid="feature-flags-button"]',
    title: 'Feature Flags',
    description: 'Manage feature toggles using Flipt.',
  },
];

export function ReactAppTour() {
  const [active, setActive] = useState(false);
  const [index, setIndex] = useState(0);

  const step = useMemo(() => TOUR_STEPS[index], [index]);

  useEffect(() => {
    try {
      const dismissed = localStorage.getItem(DISMISS_KEY) === 'true';
      const commitHash = localStorage.getItem(COMMIT_HASH_KEY);
      if (dismissed && commitHash === __APP_VERSION__) {
        return;
      }
    } catch (error) {
      logger.warn('Unable to read app tour dismissal state', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
      return;
    }
    const timer = window.setTimeout(() => {
      setActive(true);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  if (!active || !step) {
    return null;
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 2200, background: 'rgba(15, 23, 42, 0.45)' }}>
      <div
        style={{
          position: 'fixed',
          right: '24px',
          bottom: '24px',
          maxWidth: '340px',
          background: 'rgba(15, 23, 42, 0.92)',
          borderRadius: '16px',
          padding: '20px',
          color: '#f8fafc',
          boxShadow: '0 18px 42px rgba(15, 23, 42, 0.45)',
        }}
      >
        <span style={{ fontSize: '12px', textTransform: 'uppercase' }}>
          Step {index + 1} of {TOUR_STEPS.length}
        </span>
        <h2 style={{ margin: '8px 0' }}>{step.title}</h2>
        <p style={{ margin: '8px 0 14px 0' }}>{step.description}</p>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <button
            type="button"
            className="feature-flags-btn"
            onClick={() => {
              try {
                localStorage.setItem(DISMISS_KEY, 'true');
                localStorage.setItem(COMMIT_HASH_KEY, __APP_VERSION__);
              } catch (error) {
                logger.warn('Unable to persist app tour dismissal', {
                  'error.message': error instanceof Error ? error.message : String(error),
                });
              }
              setActive(false);
            }}
          >
            Skip
          </button>
          <button
            type="button"
            className="feature-flags-btn"
            onClick={() => {
              if (index >= TOUR_STEPS.length - 1) {
                try {
                  localStorage.setItem(DISMISS_KEY, 'true');
                  localStorage.setItem(COMMIT_HASH_KEY, __APP_VERSION__);
                } catch (error) {
                  logger.warn('Unable to persist app tour completion', {
                    'error.message': error instanceof Error ? error.message : String(error),
                  });
                }
                setActive(false);
                return;
              }
              setIndex((current) => current + 1);
              const target = document.querySelector(TOUR_STEPS[index + 1].selector);
              if (target instanceof HTMLElement) {
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
              }
            }}
          >
            {index === TOUR_STEPS.length - 1 ? 'Got it' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}
