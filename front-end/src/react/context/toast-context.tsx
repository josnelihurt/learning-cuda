import { createContext, useMemo, type ReactNode } from 'react';
import type { ToastContainer } from '@/components/app/toast-container';
import { logger } from '@/infrastructure/observability/otel-logger';

export type ToastApi = {
  success(title: string, message?: string, duration?: number | null): string;
  error(title: string, message?: string, duration?: number | null): string;
  warning(title: string, message?: string, duration?: number | null): string;
  info(title: string, message?: string, duration?: number | null): string;
};

export const ToastContext = createContext<ToastApi | null>(null);

function resolveHost(): ToastContainer | null {
  return document.querySelector('toast-container');
}

function createToastBridge(): ToastApi {
  return {
    success(title, message = '', duration = null) {
      const el = resolveHost();
      if (!el) {
        logger.warn('toast-container element not found; success toast skipped', {
          title,
        });
        return '';
      }
      return el.success(title, message, duration);
    },
    error(title, message = '', duration = null) {
      const el = resolveHost();
      if (!el) {
        logger.warn('toast-container element not found; error toast skipped', {
          title,
        });
        return '';
      }
      return el.error(title, message, duration);
    },
    warning(title, message = '', duration = null) {
      const el = resolveHost();
      if (!el) {
        logger.warn('toast-container element not found; warning toast skipped', {
          title,
        });
        return '';
      }
      return el.warning(title, message, duration);
    },
    info(title, message = '', duration = null) {
      const el = resolveHost();
      if (!el) {
        logger.warn('toast-container element not found; info toast skipped', {
          title,
        });
        return '';
      }
      return el.info(title, message, duration);
    },
  };
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const value = useMemo(() => createToastBridge(), []);
  return (
    <ToastContext.Provider value={value}>{children}</ToastContext.Provider>
  );
}
