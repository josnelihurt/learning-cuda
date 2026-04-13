import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ReactFrameTransportService } from './ReactFrameTransportService';

// Mock WebSocket
class MockWebSocket {
  static READY_STATE_CONNECTING = 0;
  static READY_STATE_OPEN = 1;
  static READY_STATE_CLOSING = 2;
  static READY_STATE_CLOSED = 3;

  readyState: number = MockWebSocket.READY_STATE_CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  sentMessages: any[] = [];

  constructor(public url: string) {
    // Simulate connection opening
    setTimeout(() => {
      this.readyState = MockWebSocket.READY_STATE_OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  send(data: string | Uint8Array): void {
    this.sentMessages.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.READY_STATE_CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }

  // Helper to simulate receiving a message
  simulateMessage(data: any): void {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }));
    }
  }

  // Helper to simulate error
  simulateError(error: Event): void {
    if (this.onerror) {
      this.onerror(error);
    }
  }
}

// Store original WebSocket
const OriginalWebSocket = global.WebSocket;

describe('ReactFrameTransportService', () => {
  let service: ReactFrameTransportService;
  let mockWebSocket: MockWebSocket | null = null;

  beforeEach(() => {
    // Mock global WebSocket
    global.WebSocket = MockWebSocket as any;

    // Track created WebSocket instances
    let wsInstance: MockWebSocket | null = null;
    global.WebSocket = vi.fn((url: string) => {
      wsInstance = new MockWebSocket(url);
      mockWebSocket = wsInstance;
      return wsInstance as any;
    }) as any;

    service = new ReactFrameTransportService();
  });

  afterEach(() => {
    // Clean up service
    try {
      service.close();
    } catch (e) {
      // Ignore errors during cleanup
    }

    // Restore original WebSocket
    global.WebSocket = OriginalWebSocket;
  });

  describe('initialization', () => {
    it('should initialize with WebSocket connection to /ws/frame-transport', async () => {
      await service.initialize();

      expect(global.WebSocket).toHaveBeenCalledWith(
        expect.stringContaining('/ws/frame-transport')
      );
    });

    it('should have connecting state initially', () => {
      expect(service.getConnectionStatus()).toBe('connecting');
    });

    it('should transition to connected state after WebSocket opens', async () => {
      const statusPromise = new Promise<string>((resolve) => {
        const checkStatus = () => {
          const status = service.getConnectionStatus();
          if (status === 'connected') {
            resolve(status);
          } else {
            setTimeout(checkStatus, 10);
          }
        };
        checkStatus();
      });

      await service.initialize();
      const status = await statusPromise;
      expect(status).toBe('connected');
    });

    it('should transition to failed state on connection error', async () => {
      const errorCallback = vi.fn();
      service.setErrorCallback(errorCallback);

      await service.initialize();

      // Simulate WebSocket error
      if (mockWebSocket) {
        mockWebSocket.simulateError(new Event('error'));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(service.getConnectionStatus()).toBe('failed');
      expect(errorCallback).toHaveBeenCalled();
    });
  });

  describe('sendFrame', () => {
    beforeEach(async () => {
      await service.initialize();
      // Wait for connection
      await new Promise(resolve => setTimeout(resolve, 10));
    });

    it('should send base64-encoded frame data with filter parameters', () => {
      const frameData = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD...';
      const filters = [
        { id: 'grayscale', parameters: { algorithm: 'bt601' } },
        { id: 'blur', parameters: { kernel_size: '5', sigma: '1.0' } }
      ];

      service.sendFrame(frameData, filters);

      expect(mockWebSocket?.sentMessages.length).toBeGreaterThan(0);
    });

    it('should handle empty filters array', () => {
      const frameData = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD...';
      const filters: any[] = [];

      service.sendFrame(frameData, filters);

      expect(mockWebSocket?.sentMessages.length).toBeGreaterThan(0);
    });

    it('should strip data URL prefix before sending', () => {
      const frameData = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD...';
      const filters: any[] = [];

      service.sendFrame(frameData, filters);

      const sentData = mockWebSocket?.sentMessages[0];
      if (typeof sentData === 'string') {
        expect(sentData).not.toContain('data:image/jpeg;base64,');
      }
    });
  });

  describe('onmessage handler', () => {
    beforeEach(async () => {
      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));
    });

    it('should receive WebSocketFrameResponse with processed frame', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      // Simulate receiving a processed frame response
      const mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: 'data:image/jpeg;base64,/9j/processed...'
      };

      if (mockWebSocket) {
        mockWebSocket.simulateMessage(JSON.stringify(mockResponse));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).toHaveBeenCalledWith('data:image/jpeg;base64,/9j/processed...');
    });

    it('should handle malformed messages gracefully', async () => {
      const errorCallback = vi.fn();
      service.setErrorCallback(errorCallback);

      // Send malformed JSON
      if (mockWebSocket) {
        mockWebSocket.simulateMessage('invalid json{');
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      // Should not crash, should call error callback
      expect(errorCallback).toHaveBeenCalled();
    });

    it('should ignore non-frame messages', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      // Send a different type of message
      const mockResponse = {
        type: 'ping',
        timestamp: Date.now()
      };

      if (mockWebSocket) {
        mockWebSocket.simulateMessage(JSON.stringify(mockResponse));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).not.toHaveBeenCalled();
    });
  });

  describe('frame data deserialization', () => {
    beforeEach(async () => {
      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));
    });

    it('should pass base64 data to callback without modification', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      const processedFrame = 'data:image/jpeg;base64,/9j/processed...';
      const mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: processedFrame
      };

      if (mockWebSocket) {
        mockWebSocket.simulateMessage(JSON.stringify(mockResponse));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).toHaveBeenCalledWith(processedFrame);
    });
  });

  describe('close', () => {
    it('should properly close WebSocket connection', async () => {
      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));

      service.close();

      expect(mockWebSocket?.readyState).toBe(MockWebSocket.READY_STATE_CLOSED);
    });

    it('should clear frame callback after close', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));

      service.close();

      // Try to send a message after close
      if (mockWebSocket) {
        mockWebSocket.simulateMessage(JSON.stringify({
          type: 'frame_result',
          success: true,
          processed_frame: 'data:image/jpeg;base64,test'
        }));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      // Callback should not be called after close
      expect(frameCallback).not.toHaveBeenCalled();
    });

    it('should clear error callback after close', async () => {
      const errorCallback = vi.fn();
      service.setErrorCallback(errorCallback);

      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));

      service.close();

      // Simulate error after close
      if (mockWebSocket) {
        mockWebSocket.simulateError(new Event('error'));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      // Error callback should not be called after close
      expect(errorCallback).not.toHaveBeenCalled();
    });
  });

  describe('connection state tracking', () => {
    it('should track connecting state', () => {
      expect(service.getConnectionStatus()).toBe('connecting');
    });

    it('should track connected state', async () => {
      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(service.getConnectionStatus()).toBe('connected');
    });

    it('should track disconnected state after close', async () => {
      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));

      service.close();

      expect(service.getConnectionStatus()).toBe('disconnected');
    });

    it('should track failed state on error', async () => {
      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));

      if (mockWebSocket) {
        mockWebSocket.simulateError(new Event('error'));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(service.getConnectionStatus()).toBe('failed');
    });
  });

  describe('callback registration', () => {
    beforeEach(async () => {
      await service.initialize();
      await new Promise(resolve => setTimeout(resolve, 10));
    });

    it('should register frame callback', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      const mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: 'data:image/jpeg;base64,test'
      };

      if (mockWebSocket) {
        mockWebSocket.simulateMessage(JSON.stringify(mockResponse));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).toHaveBeenCalled();
    });

    it('should register error callback', async () => {
      const errorCallback = vi.fn();
      service.setErrorCallback(errorCallback);

      if (mockWebSocket) {
        mockWebSocket.simulateError(new Event('error'));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(errorCallback).toHaveBeenCalled();
    });

    it('should allow updating frame callback', async () => {
      const firstCallback = vi.fn();
      const secondCallback = vi.fn();

      service.setFrameCallback(firstCallback);

      let mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: 'data:image/jpeg;base64,first'
      };

      if (mockWebSocket) {
        mockWebSocket.simulateMessage(JSON.stringify(mockResponse));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(firstCallback).toHaveBeenCalledTimes(1);
      expect(secondCallback).not.toHaveBeenCalled();

      // Update callback
      service.setFrameCallback(secondCallback);

      mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: 'data:image/jpeg;base64,second'
      };

      if (mockWebSocket) {
        mockWebSocket.simulateMessage(JSON.stringify(mockResponse));
      }

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(firstCallback).toHaveBeenCalledTimes(1); // Still only called once
      expect(secondCallback).toHaveBeenCalledTimes(1);
    });
  });
});
