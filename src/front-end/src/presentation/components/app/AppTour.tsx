import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from 'react';
import { logger } from '@/infrastructure/observability/otel-logger';
import styles from './AppTour.module.css';

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
    selector: 'button[data-testid="add-input-fab"], add-source-fab',
    title: 'Add Input',
    description: 'Use Add Input to bring new static images, live cameras, or videos into the workspace.',
  },
  {
    id: 'filter-panel',
    selector:
      '[data-testid="react-filters-section"], [data-testid="sidebar-column-expand"], .sidebar .filters-section, filter-panel',
    title: 'Filter Panel',
    description:
      'Drag, toggle, and fine tune filters. GPU runs filters on server GPU, while CPU keeps everything on server CPU processing.',
  },
  {
    id: 'change-image',
    selector: '[data-testid="source-card-1"], [data-testid="video-grid-host"]',
    title: 'Switch Images',
    description:
      'Select a source and use the change image control (right upper corner) to swap between available test images.',
  },
  {
    id: 'tools-dropdown',
    selector: '[data-testid="tools-dropdown-button"], tools-dropdown',
    title: 'Tools Menu',
    description: 'Open Tools to access Grafana, Jaeger, Playwright reports, and other utilities.',
  },
  {
    id: 'feature-flags',
    selector: '[data-testid="feature-flags-button"], feature-flags-button',
    title: 'Feature Flags',
    description: 'Manage feature toggles for experiments. Enable or disable flags from the built-in panel.',
  },
  {
    id: 'version-info',
    selector: '[data-testid="react-version-tooltip"] button.info-btn, version-tooltip-lit',
    title: 'Version Details',
    description: 'Click the info icon to see build details for the frontend, backend, CPP library and more.',
  },
  {
    id: 'connection-status',
    selector: '[data-testid="react-stats-panel"], [data-testid="react-stats-panel-peek"], stats-panel',
    title: 'Connection Status',
    description:
      'Monitor connection status for frame transport, gRPC, and WebRTC. Each card shows the connection state, last request, and time since last activity. Hover over the protocol name to see the full connection state.',
  },
  {
    id: 'stats-panel-toggle',
    selector: '[data-testid="react-stats-panel"], [data-testid="react-stats-panel-peek"], stats-panel',
    title: 'Stats Panel Toggle',
    description:
      'Click the stats bar to hide it, or the arrow tab at the bottom center to show it again. When collapsed, you get more workspace area.',
  },
];

export function AppTour(): ReactElement {
  const [active, setActive] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [tooltipPlacement, setTooltipPlacement] =
    useState<'right' | 'left' | 'top' | 'bottom'>('right');
  const animationFrameRef = useRef(0);
  const focusRingRef = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  const step = useMemo(() => TOUR_STEPS[stepIndex], [stepIndex]);

  const applyStyles = (element: HTMLElement | null, styles: Record<string, string>) => {
    if (!element) return;
    for (const [prop, value] of Object.entries(styles)) {
      element.style.setProperty(prop, value);
    }
  };

  const findTarget = useCallback((selector: string): HTMLElement | null => {
    const direct = document.querySelector(selector);
    if (direct instanceof HTMLElement) {
      return direct;
    }

    const body = document.body;
    if (!body) {
      return null;
    }

    const stack: (Element | ShadowRoot)[] = [body];
    const visited = new Set<Element | ShadowRoot>();

    while (stack.length > 0) {
      const node = stack.pop();
      if (!node || visited.has(node)) {
        continue;
      }
      visited.add(node);

      if (node instanceof HTMLElement && node.matches(selector)) {
        return node;
      }

      const children = Array.from(node.children ?? []);
      for (let index = children.length - 1; index >= 0; index -= 1) {
        stack.push(children[index]);
      }

      if (node instanceof Element && node.shadowRoot) {
        stack.push(node.shadowRoot);
      }
    }

    return null;
  }, []);

  const calculateTooltipPosition = useCallback(
    (rect: DOMRect, stepId?: string): { style: Record<string, string>; placement: 'right' | 'left' | 'top' | 'bottom' } => {
      const margin = 20;
      const tooltipWidth = 320;
      const tooltipHeight = 220;
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const centerY = rect.top + rect.height / 2;
      const centerX = rect.left + rect.width / 2;
      const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

      if (stepId === 'stats-panel-toggle') {
        const toggleButtonWidth = 20;
        const toggleButtonHeight = 20;
        const toggleRight = viewportWidth - 12;
        const toggleBottom = viewportHeight;
        const toggleLeft = toggleRight - toggleButtonWidth;
        const toggleTop = toggleBottom - toggleButtonHeight;
        const left = clamp(
          toggleLeft - tooltipWidth / 2 + toggleButtonWidth / 2,
          margin,
          viewportWidth - tooltipWidth - margin
        );
        const top = clamp(
          toggleTop - tooltipHeight - margin,
          margin,
          viewportHeight - tooltipHeight - margin
        );
        return {
          style: { top: `${top}px`, left: `${left}px` },
          placement: 'top',
        };
      }

      const isBottomElement = rect.top > viewportHeight / 2;
      if (isBottomElement) {
        const left = clamp(centerX - tooltipWidth / 2, margin, viewportWidth - tooltipWidth - margin);
        const top = clamp(rect.top - tooltipHeight - margin, margin, viewportHeight - tooltipHeight - margin);
        return {
          style: { top: `${top}px`, left: `${left}px` },
          placement: 'top',
        };
      }

      if (rect.right + margin + tooltipWidth <= viewportWidth) {
        const top = clamp(centerY - tooltipHeight / 2, margin, viewportHeight - tooltipHeight - margin);
        return {
          style: { top: `${top}px`, left: `${rect.right + margin}px` },
          placement: 'right',
        };
      }

      if (rect.left - margin - tooltipWidth >= 0) {
        const top = clamp(centerY - tooltipHeight / 2, margin, viewportHeight - tooltipHeight - margin);
        return {
          style: { top: `${top}px`, left: `${rect.left - tooltipWidth - margin}px` },
          placement: 'left',
        };
      }

      if (rect.bottom + margin + tooltipHeight <= viewportHeight) {
        const left = clamp(centerX - tooltipWidth / 2, margin, viewportWidth - tooltipWidth - margin);
        return {
          style: { top: `${rect.bottom + margin}px`, left: `${left}px` },
          placement: 'bottom',
        };
      }

      const left = clamp(centerX - tooltipWidth / 2, margin, viewportWidth - tooltipWidth - margin);
      const top = clamp(rect.top - tooltipHeight - margin, margin, viewportHeight - tooltipHeight - margin);
      return {
        style: { top: `${top}px`, left: `${left}px` },
        placement: 'top',
      };
    },
    []
  );

  const updateLayout = useCallback(
    (withScroll = false) => {
      const currentStep = TOUR_STEPS[stepIndex];
      if (!currentStep) {
        return;
      }

      const target = findTarget(currentStep.selector);
      if (!target) {
        if (!withScroll && focusRingRef.current?.style.width) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = requestAnimationFrame(() => updateLayout());
          return;
        }

        const fallbackRoot = findTarget('[data-testid="video-grid-host"], .sidebar, button[data-testid="add-input-fab"], body');
        const bounds = fallbackRoot?.getBoundingClientRect();
        const fallbackRect = bounds ?? new DOMRect(0, 0, window.innerWidth, window.innerHeight);
        applyStyles(focusRingRef.current, {
          top: `${fallbackRect.top}px`,
          left: `${fallbackRect.left}px`,
          width: `${fallbackRect.width}px`,
          height: `${fallbackRect.height}px`,
        });
        const { style, placement } = calculateTooltipPosition(fallbackRect, currentStep.id);
        applyStyles(tooltipRef.current, style);
        setTooltipPlacement(placement);
        return;
      }

      if (withScroll) {
        target.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }

      const rect = target.getBoundingClientRect();
      const padding = 16;
      applyStyles(focusRingRef.current, {
        top: `${Math.max(rect.top - padding, 16)}px`,
        left: `${Math.max(rect.left - padding, 16)}px`,
        width: `${rect.width + padding * 2}px`,
        height: `${rect.height + padding * 2}px`,
      });
      const { style, placement } = calculateTooltipPosition(rect, currentStep.id);
      applyStyles(tooltipRef.current, style);
      setTooltipPlacement(placement);
    },
    [calculateTooltipPosition, findTarget, stepIndex]
  );

  const dismiss = useCallback(() => {
    try {
      localStorage.setItem(DISMISS_KEY, 'true');
      localStorage.setItem(COMMIT_HASH_KEY, __APP_VERSION__);
    } catch (error) {
      logger.warn('Unable to persist app tour dismissal state', {
        'error.message': error instanceof Error ? error.message : String(error),
      });
    }
  }, []);

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

  useEffect(() => {
    if (!active) {
      return;
    }
    const onResize = () => updateLayout();
    window.addEventListener('resize', onResize);
    window.addEventListener('scroll', onResize, true);
    updateLayout(true);
    return () => {
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onResize, true);
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, [active, stepIndex, updateLayout]);

  if (!active || !step) {
    return null as unknown as ReactElement;
  }

  const arrowPlacementClass =
    tooltipPlacement === 'right'
      ? styles.arrowRight
      : tooltipPlacement === 'left'
        ? styles.arrowLeft
        : tooltipPlacement === 'bottom'
          ? styles.arrowBottom
          : styles.arrowTop;

  return (
    <div className={styles.overlay} role="presentation">
      <div ref={focusRingRef} className={styles.focusRing} />
      <div ref={tooltipRef} className={styles.tooltip} role="dialog" aria-modal="true" aria-labelledby={`${step.id}-title`}>
        <div className={`${styles.arrow} ${arrowPlacementClass}`} />
        <span className={styles.stepMeta}>
          Step {stepIndex + 1} of {TOUR_STEPS.length}
        </span>
        <h2 id={`${step.id}-title`} className={styles.title}>
          {step.title}
        </h2>
        <p className={styles.description}>
          {step.description}
        </p>
        <div className={styles.actions}>
          <button
            type="button"
            className={`${styles.buttonBase} ${styles.skipButton}`}
            onClick={() => {
              dismiss();
              setActive(false);
            }}
          >
            Skip
          </button>
          <button
            type="button"
            className={`${styles.buttonBase} ${styles.nextButton}`}
            onClick={() => {
              if (stepIndex >= TOUR_STEPS.length - 1) {
                dismiss();
                setActive(false);
                return;
              }
              setStepIndex((current) => current + 1);
            }}
          >
            {stepIndex === TOUR_STEPS.length - 1 ? 'Got it' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}
