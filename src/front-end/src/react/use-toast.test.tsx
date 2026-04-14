import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { ToastContainer } from '@/lit/components/app/toast-container';
import { ToastProvider } from './context/toast-context';
import { useToast } from './hooks/useToast';

describe('useToast / ToastProvider', () => {
  let successSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    document.body.replaceChildren();
    const mount = document.createElement('div');
    mount.id = 'react-toast-test-root';
    const toastHost = document.createElement('toast-container');
    document.body.append(mount, toastHost);
    successSpy = vi.spyOn(ToastContainer.prototype, 'success');
  });

  afterEach(() => {
    successSpy.mockRestore();
    document.body.replaceChildren();
  });

  it('invokes toast-container.success when useToast().success runs', async () => {
    function Consumer() {
      const toast = useToast();
      useEffect(() => {
        toast.success('t');
      }, [toast]);
      return null;
    }

    const rootEl = document.getElementById('react-toast-test-root');
    expect(rootEl).toBeTruthy();
    const root = createRoot(rootEl!);

    await act(async () => {
      root.render(
        <ToastProvider>
          <Consumer />
        </ToastProvider>
      );
    });

    expect(document.querySelector('toast-container')).toBeTruthy();
    await vi.waitFor(() => {
      expect(successSpy).toHaveBeenCalledWith('t', '', null);
    });

    await act(async () => {
      root.unmount();
    });
  });
});
