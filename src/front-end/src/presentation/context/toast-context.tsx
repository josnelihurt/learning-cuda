import { createContext, useMemo, useState, type ReactNode } from 'react';

import styles from './toast-context.module.css';

export type ToastApi = {
  success(title: string, message?: string, duration?: number | null): string;
  error(title: string, message?: string, duration?: number | null): string;
  warning(title: string, message?: string, duration?: number | null): string;
  info(title: string, message?: string, duration?: number | null): string;
};

export const ToastContext = createContext<ToastApi | null>(null);

type ToastKind = 'success' | 'error' | 'warning' | 'info';

type ToastItem = {
  id: string;
  kind: ToastKind;
  title: string;
  message: string;
};

function createToastApi(pushToast: (toast: Omit<ToastItem, 'id'>, duration?: number | null) => string): ToastApi {
  return {
    success(title, message = '', duration = null) {
      return pushToast({ kind: 'success', title, message }, duration);
    },
    error(title, message = '', duration = null) {
      return pushToast({ kind: 'error', title, message }, duration);
    },
    warning(title, message = '', duration = null) {
      return pushToast({ kind: 'warning', title, message }, duration);
    },
    info(title, message = '', duration = null) {
      return pushToast({ kind: 'info', title, message }, duration);
    },
  };
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const dismissToast = (id: string) => {
    setToasts((current) => current.filter((item) => item.id !== id));
  };
  const pushToast = (toast: Omit<ToastItem, 'id'>, duration: number | null = 7000): string => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    setToasts((current) => [...current, { ...toast, id }].slice(-5));
    if (duration !== null && duration > 0) {
      setTimeout(() => {
        setToasts((current) => current.filter((item) => item.id !== id));
      }, duration);
    }
    return id;
  };
  const value = useMemo(() => createToastApi(pushToast), []);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className={styles.stack}>
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={styles.card}
            data-kind={toast.kind}
            onClick={() => dismissToast(toast.id)}
          >
            <div className={styles.title}>{toast.title}</div>
            {toast.message ? <div className={styles.message}>{toast.message}</div> : null}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
