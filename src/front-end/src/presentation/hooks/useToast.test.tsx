import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { ToastProvider } from '@/presentation/context/toast-context';
import { useToast } from '@/presentation/hooks/useToast';

describe('useToast / ToastProvider', () => {
  beforeEach(() => {
    document.body.replaceChildren();
    const mount = document.createElement('div');
    mount.id = 'react-toast-test-root';
    document.body.append(mount);
  });

  afterEach(() => {
    document.body.replaceChildren();
  });

  it('invokes toast-container.success when useToast().success runs', async () => {
    function Consumer(): React.ReactNode {
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

    await vi.waitFor(() => {
      expect(document.body.textContent).toContain('t');
    });

    await act(async () => {
      root.unmount();
    });
  });
});
