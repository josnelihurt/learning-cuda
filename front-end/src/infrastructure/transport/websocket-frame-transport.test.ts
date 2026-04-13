import { describe, it, expect, vi } from 'vitest';
import { WebSocketService } from './websocket-frame-transport';
import { ImageData, FilterData } from '../../domain/value-objects';

(globalThis as any).WebSocket = { OPEN: 1 };

const makeFilters = (...ids: string[]) => ids.map((id) => new FilterData(id));
const makeStats = () => ({ updateWebSocketStatus: vi.fn() });
const makeCamera = () => ({
  setProcessing: vi.fn(),
  getLastFrameTime: vi.fn().mockReturnValue(0),
});
const makeToast = () => ({ warning: vi.fn(), error: vi.fn() });

const makeService = () => new WebSocketService(makeStats(), makeCamera(), makeToast());

describe('WebSocketService', () => {
  it('sends frames when websocket is connected', () => {
    const mockWs = { readyState: 1, send: vi.fn() };
    const service = makeService();
    (service as any).ws = mockWs;

    service.sendFrame('data:image/png;base64,test', 32, 32, makeFilters('grayscale'), 'gpu');

    expect(mockWs.send).toHaveBeenCalled();
  });

  it('does nothing when websocket is disconnected', () => {
    const mockWs = { readyState: 3, send: vi.fn() };
    const service = makeService();
    (service as any).ws = mockWs;

    service.sendFrame('data:image/png;base64,test', 32, 32, makeFilters('none'), 'gpu');

    expect(mockWs.send).not.toHaveBeenCalled();
  });

  it('validates image payload before sending', () => {
    const service = makeService();

    expect(() =>
      service.sendFrameWithImageData(new ImageData('', 10, 10), makeFilters(), 'gpu')
    ).toThrow('Image data cannot be empty');

    expect(() =>
      service.sendFrameWithImageData(new ImageData('data:image/png;base64,test', -1, 10), makeFilters(), 'gpu')
    ).toThrow('Image dimensions must be positive');
  });

  it('starts video playback using provided filters', () => {
    const mockWs = { readyState: 1, send: vi.fn() };
    const service = makeService();
    (service as any).ws = mockWs;

    service.sendStartVideo('video-123', makeFilters('grayscale', 'blur'), 'gpu');

    expect(mockWs.send).toHaveBeenCalled();
  });
});
