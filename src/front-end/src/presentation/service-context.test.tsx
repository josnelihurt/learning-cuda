import { describe, it, expect, vi, afterEach } from 'vitest';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { useServiceContext } from './context/service-context';
import { renderWithService } from './test-utils/render-with-service';
import type { GrpcClients } from './context/service-context';

function Consumer() {
  useServiceContext();
  return <span>has-clients</span>;
}

afterEach(() => {
  document.body.replaceChildren();
});

describe('ServiceContext / useServiceContext', () => {
  it('throws when used outside provider', () => {
    function Bad() {
      useServiceContext();
      return null;
    }
    const div = document.createElement('div');
    document.body.appendChild(div);
    const root = createRoot(div);
    expect(() => {
      act(() => root.render(<Bad />));
    }).toThrow(/useServiceContext/);
    act(() => root.unmount());
    div.remove();
  });

  it('returns clients when wrapped with renderWithService', () => {
    const imageProcessorClient = {
      listFilters: vi.fn(),
    } as unknown as GrpcClients['imageProcessorClient'];
    const remoteManagementClient = {} as GrpcClients['remoteManagementClient'];

    const { unmount } = renderWithService(<Consumer />, {
      imageProcessorClient,
      remoteManagementClient,
    });

    expect(document.body.textContent).toContain('has-clients');
    unmount();
  });
});
