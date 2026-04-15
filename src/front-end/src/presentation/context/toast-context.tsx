import { createContext, useMemo, useState, type ReactNode } from 'react';

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

function getColor(kind: ToastKind): string {
  if (kind === 'success') return '#4caf50';
  if (kind === 'error') return '#f44336';
  if (kind === 'warning') return '#ff9800';
  return '#2196f3';
}

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
      <div
        style={{
          position: 'fixed',
          top: '20px',
          right: '20px',
          zIndex: 10000,
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
        }}
      >
        {toasts.map((toast) => (
          <div
            key={toast.id}
            style={{
              minWidth: '300px',
              maxWidth: '400px',
              padding: '14px 16px',
              borderRadius: '8px',
              borderLeft: `4px solid ${getColor(toast.kind)}`,
              background: '#fff',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            }}
          >
            <div style={{ fontWeight: 600 }}>{toast.title}</div>
            {toast.message ? <div style={{ color: '#666' }}>{toast.message}</div> : null}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
