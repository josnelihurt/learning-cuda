/**
 * Unit tests for ReactWebRTCService
 * TDD RED phase - These tests should fail initially
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ReactWebRTCService } from './ReactWebRTCService';

// Mock WebRTC APIs
const mockPeerConnection = {
  close: vi.fn(),
  createDataChannel: vi.fn(),
  createOffer: vi.fn(),
  setLocalDescription: vi.fn(),
  setRemoteDescription: vi.fn(),
  addIceCandidate: vi.fn(),
  onicecandidate: null,
  onconnectionstatechange: null,
  oniceconnectionstatechange: null,
};

const mockDataChannel = {
  send: vi.fn(),
  close: vi.fn(),
  readyState: 'open',
  onopen: null,
  onmessage: null,
  onerror: null,
  onclose: null,
};

const mockWebSocket = {
  send: vi.fn(),
  close: vi.fn(),
  readyState: 1 as any, // WebSocket.OPEN = 1
  onopen: null,
  onmessage: null,
  onerror: null,
  onclose: null,
};

describe('ReactWebRTCService', () => {
  let service: ReactWebRTCService;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Mock RTCPeerConnection
    global.RTCPeerConnection = vi.fn(() => mockPeerConnection) as any;

    // Mock RTCDataChannel
    (mockPeerConnection.createDataChannel as any).mockReturnValue(mockDataChannel);

    // Mock WebSocket
    global.WebSocket = vi.fn(() => mockWebSocket) as any;

    // Mock navigator.mediaDevices
    global.navigator = {
      mediaDevices: {
        getUserMedia: vi.fn(),
      },
    } as any;

    service = new ReactWebRTCService();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Test 1: initialize() should check browser support and set initialized flag', () => {
    it('should initialize successfully when WebRTC is supported', async () => {
      await service.initialize();

      expect(service.isInitialized()).toBe(true);
    });

    it('should not initialize when WebRTC is not supported', async () => {
      // Make WebRTC unsupported
      (global.RTCPeerConnection as any) = undefined;

      const service2 = new ReactWebRTCService();
      await service2.initialize();

      expect(service2.isInitialized()).toBe(false);
    });
  });

  describe('Test 2: createPeerConnection() should create RTCPeerConnection with STUN server', () => {
    it('should create peer connection with STUN server', async () => {
      await service.initialize();

      const pc = service.createPeerConnection();

      expect(pc).not.toBeNull();
      expect(global.RTCPeerConnection).toHaveBeenCalledWith(
        expect.objectContaining({
          iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
          ],
        })
      );
    });

    it('should return null when WebRTC is not supported', () => {
      (global.RTCPeerConnection as any) = undefined;
      const service2 = new ReactWebRTCService();

      const pc = service2.createPeerConnection();

      expect(pc).toBeNull();
    });
  });

  describe('Test 3: createDataChannel() should create RTCDataChannel with specified label', () => {
    it('should create data channel with correct label', async () => {
      await service.initialize();

      const pc = service.createPeerConnection()!;
      const dc = service.createDataChannel(pc, 'test-channel');

      expect(dc).not.toBeNull();
      expect(mockPeerConnection.createDataChannel).toHaveBeenCalledWith(
        'test-channel',
        undefined
      );
    });

    it('should return null when peer connection is not provided', async () => {
      await service.initialize();

      const dc = service.createDataChannel(null as any, 'test-channel');

      expect(dc).toBeNull();
    });
  });

  describe('Test 4: createSession() should establish full session lifecycle', () => {
    it('should establish session with peer connection, data channel, WebSocket signaling, ICE exchange, and heartbeat', async () => {
      // Mock createOffer to return SDP offer
      mockPeerConnection.createOffer.mockResolvedValue({
        type: 'offer',
        sdp: 'mock-sdp-offer',
      });

      // Mock WebSocket onopen to resolve connection
      const wsOpenPromise = new Promise<void>((resolve) => {
        mockWebSocket.onopen = () => resolve();
      });

      // Simulate WebSocket connection
      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen({} as Event);
        }
      }, 10);

      await wsOpenPromise;

      // Simulate data channel opening
      setTimeout(() => {
        if (mockDataChannel.onopen) {
          mockDataChannel.onopen({} as Event);
        }
      }, 20);

      const session = await service.createSession('test-source');

      expect(session).not.toBeNull();
      expect(session.getId()).toBeDefined();
      expect(mockPeerConnection.createDataChannel).toHaveBeenCalledWith(
        'ping-pong-channel',
        undefined
      );
      expect(mockPeerConnection.createOffer).toHaveBeenCalled();
      expect(mockPeerConnection.setLocalDescription).toHaveBeenCalled();
      expect(mockWebSocket.send).toHaveBeenCalled();
    });

    it('should throw error when WebRTC is not supported', async () => {
      (global.RTCPeerConnection as any) = undefined;
      const service2 = new ReactWebRTCService();

      await expect(service2.createSession('test-source')).rejects.toThrow(
        'WebRTC is not supported in this browser'
      );
    });
  });

  describe('Test 5: closeSession() should cleanup all resources', () => {
    it('should stop heartbeat, close WebSocket, close data channel, close peer connection', async () => {
      // Mock createOffer
      mockPeerConnection.createOffer.mockResolvedValue({
        type: 'offer',
        sdp: 'mock-sdp-offer',
      });

      // Mock WebSocket onopen
      const wsOpenPromise = new Promise<void>((resolve) => {
        mockWebSocket.onopen = () => resolve();
      });

      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen({} as Event);
        }
      }, 10);

      await wsOpenPromise;

      // Simulate data channel opening
      setTimeout(() => {
        if (mockDataChannel.onopen) {
          mockDataChannel.onopen({} as Event);
        }
      }, 20);

      const session = await service.createSession('test-source');
      const sessionId = session.getId();

      await service.closeSession(sessionId);

      expect(mockWebSocket.send).toHaveBeenCalled(); // closeSession message
      expect(mockWebSocket.close).toHaveBeenCalled();
      expect(mockPeerConnection.close).toHaveBeenCalled();
    });
  });

  describe('Test 6: getConnectionStatus() should return connection state', () => {
    it('should return disconnected state initially', () => {
      const status = service.getConnectionStatus();

      expect(status.state).toBe('disconnected');
      expect(status.lastRequest).toBeNull();
      expect(status.lastRequestTime).toBeNull();
    });

    it('should return connected state when session is active', async () => {
      // Mock createOffer
      mockPeerConnection.createOffer.mockResolvedValue({
        type: 'offer',
        sdp: 'mock-sdp-offer',
      });

      // Mock WebSocket onopen
      const wsOpenPromise = new Promise<void>((resolve) => {
        mockWebSocket.onopen = () => resolve();
      });

      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen({} as Event);
        }
      }, 10);

      await wsOpenPromise;

      // Simulate data channel opening
      setTimeout(() => {
        if (mockDataChannel.onopen) {
          mockDataChannel.onopen({} as Event);
        }
      }, 20);

      // Simulate heartbeat pong
      setTimeout(() => {
        if (mockDataChannel.onmessage) {
          mockDataChannel.onmessage({ data: 'pong' } as MessageEvent);
        }
      }, 30);

      await service.createSession('test-source');

      // Wait for heartbeat to update state
      await new Promise(resolve => setTimeout(resolve, 50));

      const status = service.getConnectionStatus();

      expect(status.state).toBe('connected');
      expect(status.lastRequest).toBe('Heartbeat pong');
      expect(status.lastRequestTime).not.toBeNull();
    });
  });

  describe('Test 7: Heartbeat should send ping every 5s and expect pong', () => {
    it('should send ping on heartbeat interval', async () => {
      // Mock createOffer
      mockPeerConnection.createOffer.mockResolvedValue({
        type: 'offer',
        sdp: 'mock-sdp-offer',
      });

      // Mock WebSocket onopen
      const wsOpenPromise = new Promise<void>((resolve) => {
        mockWebSocket.onopen = () => resolve();
      });

      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen({} as Event);
        }
      }, 10);

      await wsOpenPromise;

      // Simulate data channel opening
      setTimeout(() => {
        if (mockDataChannel.onopen) {
          mockDataChannel.onopen({} as Event);
        }
      }, 20);

      await service.createSession('test-source');

      // Wait for heartbeat to send ping
      await new Promise(resolve => setTimeout(resolve, 50));

      expect(mockDataChannel.send).toHaveBeenCalledWith('ping');
    });

    it('should update lastRequest and lastRequestTime on pong', async () => {
      // Mock createOffer
      mockPeerConnection.createOffer.mockResolvedValue({
        type: 'offer',
        sdp: 'mock-sdp-offer',
      });

      // Mock WebSocket onopen
      const wsOpenPromise = new Promise<void>((resolve) => {
        mockWebSocket.onopen = () => resolve();
      });

      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen({} as Event);
        }
      }, 10);

      await wsOpenPromise;

      // Simulate data channel opening
      setTimeout(() => {
        if (mockDataChannel.onopen) {
          mockDataChannel.onopen({} as Event);
        }
      }, 20);

      await service.createSession('test-source');

      // Simulate heartbeat pong
      setTimeout(() => {
        if (mockDataChannel.onmessage) {
          mockDataChannel.onmessage({ data: 'pong' } as MessageEvent);
        }
      }, 30);

      // Wait for pong to be processed
      await new Promise(resolve => setTimeout(resolve, 50));

      const status = service.getConnectionStatus();

      expect(status.lastRequest).toBe('Heartbeat pong');
      expect(status.lastRequestTime).not.toBeNull();
    });
  });

  describe('Test 8: WebSocket signaling should exchange messages', () => {
    it('should send startSession message via WebSocket', async () => {
      // Mock createOffer
      mockPeerConnection.createOffer.mockResolvedValue({
        type: 'offer',
        sdp: 'mock-sdp-offer',
      });

      // Mock WebSocket onopen
      const wsOpenPromise = new Promise<void>((resolve) => {
        mockWebSocket.onopen = () => resolve();
      });

      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen({} as Event);
        }
      }, 10);

      await wsOpenPromise;

      await service.createSession('test-source');

      expect(mockWebSocket.send).toHaveBeenCalled();
      const sentMessage = JSON.parse(mockWebSocket.send.mock.calls[0][0]);
      expect(sentMessage.message.case).toBe('startSession');
    });

    it('should handle ICE candidate messages', async () => {
      // Mock createOffer
      mockPeerConnection.createOffer.mockResolvedValue({
        type: 'offer',
        sdp: 'mock-sdp-offer',
      });

      // Mock WebSocket onopen
      const wsOpenPromise = new Promise<void>((resolve) => {
        mockWebSocket.onopen = () => resolve();
      });

      setTimeout(() => {
        if (mockWebSocket.onopen) {
          mockWebSocket.onopen({} as Event);
        }
      }, 10);

      await wsOpenPromise;

      const session = await service.createSession('test-source');

      // Simulate ICE candidate event
      const iceEvent = {
        candidate: {
          candidate: 'mock-candidate',
          sdpMid: '0',
          sdpMLineIndex: 0,
        },
      } as RTCPeerConnectionIceEvent;

      if (mockPeerConnection.onicecandidate) {
        mockPeerConnection.onicecandidate(iceEvent);
      }

      expect(mockWebSocket.send).toHaveBeenCalled();
    });
  });

  describe('Browser support detection', () => {
    it('isSupported() should return true when all APIs are available', () => {
      expect(service.isSupported()).toBe(true);
    });

    it('isSupported() should return false when RTCPeerConnection is missing', () => {
      (global.RTCPeerConnection as any) = undefined;
      const service2 = new ReactWebRTCService();

      expect(service2.isSupported()).toBe(false);
    });

    it('isSupported() should return false when getUserMedia is missing', () => {
      (global.navigator as any).mediaDevices.getUserMedia = undefined;
      const service2 = new ReactWebRTCService();

      expect(service2.isSupported()).toBe(false);
    });
  });
});
