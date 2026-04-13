import { vi } from 'vitest';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('./services/otel-logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    initialize: vi.fn(),
    isDebugEnabled: vi.fn().mockReturnValue(false),
    shutdown: vi.fn(),
  },
}));

// WebRTC API stubs for React test imports (Phase 4 will replace with typed mocks)
global.RTCPeerConnection = vi.fn() as any;
global.RTCPeerConnection.prototype = {
  createDataChannel: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  createOffer: vi.fn(),
  createAnswer: vi.fn(),
  setLocalDescription: vi.fn(),
  setRemoteDescription: vi.fn(),
  addTrack: vi.fn(),
} as any;

if (!navigator.mediaDevices) {
  Object.defineProperty(navigator, 'mediaDevices', {
    value: {
      getUserMedia: vi.fn(),
      enumerateDevices: vi.fn(),
    },
    writable: true,
  });
}
