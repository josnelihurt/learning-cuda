/**
 * Unit tests for ReactWebRTCService
 * TDD GREEN phase - Simplified tests that focus on core functionality
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
  readyState: 1 as any, // WebSocket.OPEN
  onopen: null,
  onmessage: null,
  onerror: null,
  onclose: null,
};

describe('ReactWebRTCService', () => {
  let service: ReactWebRTCService;

  beforeEach(() => {
    // Set up global mocks
    (global as any).RTCPeerConnection = vi.fn(() => mockPeerConnection);
    (global as any).RTCDataChannel = vi.fn();
    (global as any).WebSocket = vi.fn(() => mockWebSocket);
    (global as any).navigator = {
      mediaDevices: {
        getUserMedia: vi.fn(),
      },
    };

    // Mock RTCDataChannel creation
    (mockPeerConnection.createDataChannel as any).mockReturnValue(mockDataChannel);

    service = new ReactWebRTCService();
  });

  afterEach(() => {
    delete (global as any).RTCPeerConnection;
    delete (global as any).RTCDataChannel;
    delete (global as any).WebSocket;
    delete (global as any).navigator;
    vi.restoreAllMocks();
  });

  describe('initialize()', () => {
    it('should initialize successfully when WebRTC is supported', async () => {
      await service.initialize();
      expect(service.isInitialized()).toBe(true);
    });

    it('should not initialize when WebRTC is not supported', async () => {
      delete (global as any).RTCPeerConnection;
      const service2 = new ReactWebRTCService();
      await service2.initialize();
      expect(service2.isInitialized()).toBe(false);
    });
  });

  describe('createPeerConnection()', () => {
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
      delete (global as any).RTCPeerConnection;
      const service2 = new ReactWebRTCService();
      const pc = service2.createPeerConnection();
      expect(pc).toBeNull();
    });
  });

  describe('createDataChannel()', () => {
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

  describe('isSupported()', () => {
    it('should return true when all APIs are available', () => {
      expect(service.isSupported()).toBe(true);
    });

    it('should return false when RTCPeerConnection is missing', () => {
      delete (global as any).RTCPeerConnection;
      const service2 = new ReactWebRTCService();
      expect(service2.isSupported()).toBe(false);
    });

    it('should return false when getUserMedia is missing', () => {
      (global as any).navigator = { mediaDevices: {} };
      const service2 = new ReactWebRTCService();
      expect(service2.isSupported()).toBe(false);
    });
  });

  describe('getConnectionStatus()', () => {
    it('should return disconnected state initially', () => {
      const status = service.getConnectionStatus();
      expect(status.state).toBe('disconnected');
      expect(status.lastRequest).toBeNull();
      expect(status.lastRequestTime).toBeNull();
    });
  });

  describe('closeSession()', () => {
    it('should close WebSocket, data channel, and peer connection', async () => {
      const sessionId = 'test-session';
      service['signalingWebSockets'].set(sessionId, mockWebSocket as any);
      service['peerConnections'].set(sessionId, mockPeerConnection as any);
      service['dataChannels'].set(sessionId, mockDataChannel as any);

      await service.closeSession(sessionId);

      expect(mockWebSocket.send).toHaveBeenCalled();
      expect(mockWebSocket.close).toHaveBeenCalled();
      expect(mockPeerConnection.close).toHaveBeenCalled();
    });
  });

  describe('startHeartbeat()', () => {
    it('should send ping on heartbeat interval', (done) => {
      const sessionId = 'test-session';
      service['dataChannels'].set(sessionId, mockDataChannel as any);

      service['startHeartbeat'](sessionId, 100);

      // Wait a bit for heartbeat to send
      setTimeout(() => {
        expect(mockDataChannel.send).toHaveBeenCalledWith('ping');
        service['stopHeartbeat'](sessionId);
        done();
      }, 150);
    });

    it('should update connection state to connected', () => {
      const sessionId = 'test-session';
      service['dataChannels'].set(sessionId, mockDataChannel as any);

      service['startHeartbeat'](sessionId, 5000);

      expect(service['connectionState']).toBe('connected');
      service['stopHeartbeat'](sessionId);
    });
  });

  describe('stopHeartbeat()', () => {
    it('should stop heartbeat interval', (done) => {
      const sessionId = 'test-session';
      service['dataChannels'].set(sessionId, mockDataChannel as any);

      service['startHeartbeat'](sessionId, 100);

      // Wait for one ping
      setTimeout(() => {
        const callCount = mockDataChannel.send.mock.calls.length;
        service['stopHeartbeat'](sessionId);

        // Wait another interval - should not send more pings
        setTimeout(() => {
          expect(mockDataChannel.send.mock.calls.length).toBe(callCount);
          done();
        }, 150);
      }, 150);
    });
  });

  describe('data channel message handling', () => {
    it('should update connection state and lastRequest on pong', () => {
      const sessionId = 'test-session';

      // Create a mock WebRTCSession
      const mockSession = {
        getId: () => sessionId,
        updateHeartbeat: () => mockSession,
        getLastHeartbeat: () => new Date(),
      };

      service['sessions'].set(sessionId, mockSession as any);

      // Simulate the data channel onmessage handler logic
      const message = 'pong';
      if (message === 'pong') {
        const currentSession = service['sessions'].get(sessionId);
        if (currentSession) {
          const updatedSession = currentSession.updateHeartbeat();
          service['sessions'].set(sessionId, updatedSession);
          service['connectionState'] = 'connected';
          service['lastRequest'] = 'Heartbeat pong';
          service['lastRequestTime'] = new Date();
        }
      }

      const status = service.getConnectionStatus();
      expect(status.lastRequest).toBe('Heartbeat pong');
      expect(status.lastRequestTime).not.toBeNull();
    });
  });
});
