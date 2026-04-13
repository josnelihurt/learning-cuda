import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ReactFrameTransportService } from './ReactFrameTransportService';

// Simple mock WebSocket class
let mockWebSocketInstance: any = null;

class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState: number = MockWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  sentMessages: any[] = [];

  constructor(public url: string) {
    mockWebSocketInstance = this;

    // Simulate async connection
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  send(data: string | Uint8Array): void {
    this.sentMessages.push(data);
  }

  close(): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }

  // Test helpers
  simulateMessage(data: any): void {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }));
    }
  }

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

  const setupService = async () => {
    mockWebSocketInstance = null;
    (global as any).WebSocket = MockWebSocket;
    service = new ReactFrameTransportService();
    await service.initialize();
    await new Promise(resolve => setTimeout(resolve, 50));
    // Force WebSocket to open state for testing
    if (mockWebSocketInstance && mockWebSocketInstance.readyState !== MockWebSocket.OPEN) {
      mockWebSocketInstance.readyState = MockWebSocket.OPEN;
    }
  };

  const cleanupService = () => {
    try {
      service.close();
    } catch (e) {
      // Ignore
    }
    mockWebSocketInstance = null;
    (global as any).WebSocket = OriginalWebSocket;
  };

  beforeEach(async () => {
    await setupService();
  });

  afterEach(() => {
    cleanupService();
  });

  describe('initialization', () => {
    it('should initialize with WebSocket connection to /ws/frame-transport', async () => {
      await setupService();
      expect(mockWebSocketInstance).toBeTruthy();
      expect(mockWebSocketInstance.url).toContain('/ws/frame-transport');
      cleanupService();
    });

    it('should have connecting state initially', async () => {
      mockWebSocketInstance = null;
      (global as any).WebSocket = MockWebSocket;
      service = new ReactFrameTransportService();
      expect(service.getConnectionStatus()).toBe('connecting');
      cleanupService();
    });

    it('should transition to connected state after WebSocket opens', async () => {
      await setupService();
      expect(service.getConnectionStatus()).toBe('connected');
      cleanupService();
    });

    it('should transition to failed state on connection error', async () => {
      mockWebSocketInstance = null;
      (global as any).WebSocket = MockWebSocket;
      service = new ReactFrameTransportService();

      const errorCallback = vi.fn();
      service.setErrorCallback(errorCallback);

      // Mock initialize to fail
      const originalInit = service.initialize;
      service.initialize = async () => {
        throw new Error('Connection failed');
      };

      try {
        await service.initialize();
      } catch (e) {
        // Expected
      }

      expect(service.getConnectionStatus()).toBe('failed');
      expect(errorCallback).toHaveBeenCalled();
      cleanupService();
    });
  });

  describe('sendFrame', () => {
    it('should send base64-encoded frame data with filter parameters', () => {
      const frameData = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD...';
      const filters = [
        { id: 'grayscale', parameters: { algorithm: 'bt601' } },
        { id: 'blur', parameters: { kernel_size: '5', sigma: '1.0' } }
      ];

      service.sendFrame(frameData, filters);

      expect(mockWebSocketInstance?.sentMessages.length).toBeGreaterThan(0);
    });

    it('should handle empty filters array', () => {
      const frameData = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD...';
      const filters: any[] = [];

      service.sendFrame(frameData, filters);

      expect(mockWebSocketInstance?.sentMessages.length).toBeGreaterThan(0);
    });

    it('should strip data URL prefix before sending', () => {
      const frameData = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD...';
      const filters: any[] = [];

      service.sendFrame(frameData, filters);

      const sentData = mockWebSocketInstance?.sentMessages[0];
      if (typeof sentData === 'string') {
        expect(sentData).not.toContain('data:image/jpeg;base64,');
      }
    });
  });

  describe('onmessage handler', () => {
    it('should receive WebSocketFrameResponse with processed frame', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      const mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: 'data:image/jpeg;base64,/9j/processed...'
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(mockResponse));
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).toHaveBeenCalledWith('data:image/jpeg;base64,/9j/processed...');
    });

    it('should handle malformed messages gracefully', async () => {
      const errorCallback = vi.fn();
      service.setErrorCallback(errorCallback);

      mockWebSocketInstance?.simulateMessage('invalid json{');
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(errorCallback).toHaveBeenCalled();
    });

    it('should ignore non-frame messages', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      const mockResponse = {
        type: 'ping',
        timestamp: Date.now()
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(mockResponse));
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).not.toHaveBeenCalled();
    });
  });

  describe('frame data deserialization', () => {
    it('should pass base64 data to callback without modification', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      const processedFrame = 'data:image/jpeg;base64,/9j/processed...';
      const mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: processedFrame
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(mockResponse));
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).toHaveBeenCalledWith(processedFrame);
    });
  });

  describe('close', () => {
    it('should properly close WebSocket connection', () => {
      service.close();

      expect(mockWebSocketInstance?.readyState).toBe(MockWebSocket.CLOSED);
    });

    it('should clear frame callback after close', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      service.close();

      mockWebSocketInstance?.simulateMessage(JSON.stringify({
        type: 'frame_result',
        success: true,
        processed_frame: 'data:image/jpeg;base64,test'
      }));

      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).not.toHaveBeenCalled();
    });

    it('should clear error callback after close', async () => {
      const errorCallback = vi.fn();
      service.setErrorCallback(errorCallback);

      service.close();

      mockWebSocketInstance?.simulateError(new Event('error'));
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(errorCallback).not.toHaveBeenCalled();
    });
  });

  describe('connection state tracking', () => {
    it('should track connecting state', async () => {
      mockWebSocketInstance = null;
      (global as any).WebSocket = MockWebSocket;
      service = new ReactFrameTransportService();
      expect(service.getConnectionStatus()).toBe('connecting');
      cleanupService();
    });

    it('should track connected state', () => {
      expect(service.getConnectionStatus()).toBe('connected');
    });

    it('should track disconnected state after close', () => {
      service.close();

      expect(service.getConnectionStatus()).toBe('disconnected');
    });

    it('should track failed state on error', async () => {
      mockWebSocketInstance?.simulateError(new Event('error'));
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(service.getConnectionStatus()).toBe('failed');
    });
  });

  describe('callback registration', () => {
    it('should register frame callback', async () => {
      const frameCallback = vi.fn();
      service.setFrameCallback(frameCallback);

      const mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: 'data:image/jpeg;base64,test'
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(mockResponse));
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(frameCallback).toHaveBeenCalled();
    });

    it('should register error callback', async () => {
      const errorCallback = vi.fn();
      service.setErrorCallback(errorCallback);

      mockWebSocketInstance?.simulateError(new Event('error'));
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

      mockWebSocketInstance?.simulateMessage(JSON.stringify(mockResponse));
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(firstCallback).toHaveBeenCalledTimes(1);
      expect(secondCallback).not.toHaveBeenCalled();

      service.setFrameCallback(secondCallback);

      mockResponse = {
        type: 'frame_result',
        success: true,
        processed_frame: 'data:image/jpeg;base64,second'
      };

      mockWebSocketInstance?.simulateMessage(JSON.stringify(mockResponse));
      await new Promise(resolve => setTimeout(resolve, 10));

      expect(firstCallback).toHaveBeenCalledTimes(1);
      expect(secondCallback).toHaveBeenCalledTimes(1);
    });
  });
});
