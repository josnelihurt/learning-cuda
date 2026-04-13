import {
  ProcessImageRequest,
  GenericFilterSelection,
  GenericFilterParameterSelection,
  WebSocketFrameRequest,
  TraceContext,
  AcceleratorType,
} from '@/gen/image_processor_service_pb';
import { propagation, context } from '@opentelemetry/api';
import type { ActiveFilterState } from '@/react/components/filters/FilterPanel';

/**
 * ReactFrameTransportService handles WebSocket-based frame transport for video streaming.
 *
 * Manages:
 * - WebSocket connection to /ws/frame-transport endpoint
 * - Sending frames to backend with filter parameters
 * - Receiving processed frames from backend
 * - Connection state tracking
 * - Callback registration for frame and error events
 *
 * Adapted from Lit WebSocketService for React compatibility.
 */
export class ReactFrameTransportService {
  private ws: WebSocket | null = null;
  private connectionState: 'connecting' | 'connected' | 'disconnected' | 'failed' = 'connecting';
  private frameCallback: ((frameData: string) => void) | null = null;
  private errorCallback: ((error: Error) => void) | null = null;
  private wsUrl: string;

  /**
   * Creates a new ReactFrameTransportService instance.
   *
   * @param wsUrl - Optional WebSocket URL (defaults to /ws/frame-transport with auto-detect protocol)
   */
  constructor(wsUrl?: string) {
    if (wsUrl) {
      this.wsUrl = wsUrl;
    } else {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      this.wsUrl = `${protocol}//${window.location.host}/ws/frame-transport`;
    }
  }

  /**
   * Initializes the WebSocket connection.
   *
   * @returns Promise that resolves when connection is established
   */
  async initialize(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.connectionState = 'connecting';
        this.ws = new WebSocket(this.wsUrl);

        this.ws.onopen = () => {
          this.connectionState = 'connected';
          resolve();
        };

        this.ws.onerror = (error) => {
          this.handleError(error);
          reject(new Error('WebSocket connection failed'));
        };

        this.ws.onclose = () => {
          this.handleClose();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };
      } catch (error) {
        this.connectionState = 'failed';
        if (this.errorCallback) {
          this.errorCallback(error instanceof Error ? error : new Error(String(error)));
        }
        reject(error);
      }
    });
  }

  /**
   * Sends a frame to the backend with filter parameters.
   *
   * @param base64Data - Base64-encoded frame data (with or without data URL prefix)
   * @param filters - Array of active filter states to apply
   */
  sendFrame(base64Data: string, filters: ActiveFilterState[]): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      const error = new Error('WebSocket not connected');
      if (this.errorCallback) {
        this.errorCallback(error);
      }
      return;
    }

    try {
      // Strip data URL prefix if present
      const imageDataB64 = base64Data.replace(/^data:image\/(png|jpeg|webp);base64,/, '');
      const imageBytes = Uint8Array.from(atob(imageDataB64), (c) => c.charCodeAt(0));

      // Convert ActiveFilterState[] to GenericFilterSelection[]
      const genericFilters = this.convertFiltersToProtobuf(filters);

      // Create ProcessImageRequest
      const request = new ProcessImageRequest({
        imageData: imageBytes,
        width: 640, // Default width (should be configurable in future)
        height: 480, // Default height (should be configurable in future)
        channels: 3, // RGB
        accelerator: AcceleratorType.CUDA, // Default to CUDA
        genericFilters,
      });

      // Add OpenTelemetry trace context
      const carrier: { [key: string]: string } = {};
      propagation.inject(context.active(), carrier);

      const traceContext = new TraceContext({
        traceparent: carrier['traceparent'] || '',
        tracestate: carrier['tracestate'] || '',
      });

      // Wrap in WebSocketFrameRequest
      const frameRequest = new WebSocketFrameRequest({
        type: 'frame',
        request: request,
        traceContext: traceContext,
      });

      // Send as JSON (binary format can be added later)
      this.ws.send(frameRequest.toJsonString());
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      if (this.errorCallback) {
        this.errorCallback(err);
      }
    }
  }

  /**
   * Registers a callback to receive processed frames.
   *
   * @param callback - Function to call when a processed frame is received
   */
  setFrameCallback(callback: ((frameData: string) => void) | null): void {
    this.frameCallback = callback;
  }

  /**
   * Registers a callback to receive errors.
   *
   * @param callback - Function to call when an error occurs
   */
  setErrorCallback(callback: ((error: Error) => void) | null): void {
    this.errorCallback = callback;
  }

  /**
   * Gets the current connection status.
   *
   * @returns Connection state string
   */
  getConnectionStatus(): string {
    return this.connectionState;
  }

  /**
   * Closes the WebSocket connection and clears all callbacks.
   */
  close(): void {
    // Clear callbacks
    this.frameCallback = null;
    this.errorCallback = null;

    // Close WebSocket
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
      this.ws.onmessage = null;
      this.ws.close();
      this.ws = null;
    }

    // Update state
    this.connectionState = 'disconnected';
  }

  /**
   * Handles WebSocket open event (internal).
   */
  private handleOpen(): void {
    this.connectionState = 'connected';
  }

  /**
   * Handles WebSocket error event (internal).
   *
   * @param error - Error event
   */
  private handleError(error: Event): void {
    this.connectionState = 'failed';
    const err = new Error('WebSocket error occurred');
    if (this.errorCallback) {
      this.errorCallback(err);
    }
  }

  /**
   * Handles WebSocket close event (internal).
   */
  private handleClose(): void {
    this.connectionState = 'disconnected';
  }

  /**
   * Handles WebSocket message event (internal).
   *
   * @param event - Message event
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);

      // Check if this is a frame result message
      if (data.type === 'frame_result' || data.type === 'video_frame') {
        if (data.success && data.processed_frame) {
          // Call frame callback with processed frame data
          if (this.frameCallback) {
            this.frameCallback(data.processed_frame);
          }
        } else if (!data.success && this.errorCallback) {
          // Frame processing failed
          this.errorCallback(new Error(data.error || 'Frame processing failed'));
        }
      }
      // Ignore other message types (ping, status, etc.)
    } catch (error) {
      // Failed to parse message
      const err = new Error('Failed to parse WebSocket message');
      if (this.errorCallback) {
        this.errorCallback(err);
      }
    }
  }

  /**
   * Converts ActiveFilterState[] to GenericFilterSelection[] (internal).
   *
   * @param filters - Array of active filter states
   * @returns Array of GenericFilterSelection protobuf messages
   */
  private convertFiltersToProtobuf(filters: ActiveFilterState[]): GenericFilterSelection[] {
    return filters.map((filter) => {
      const parameterSelections = Object.entries(filter.parameters)
        .map(([parameterId, value]) => {
          const values = this.serializeParameterValue(value);
          if (values.length === 0) {
            return null;
          }
          return new GenericFilterParameterSelection({
            parameterId,
            values,
          });
        })
        .filter((selection): selection is GenericFilterParameterSelection => selection !== null);

      return new GenericFilterSelection({
        filterId: filter.id,
        parameters: parameterSelections,
      });
    });
  }

  /**
   * Serializes a single parameter value to string array (internal).
   *
   * @param value - Parameter value to serialize
   * @returns Array of string values
   */
  private serializeParameterValue(value: string): string[] {
    if (!value) {
      return [];
    }
    return [value];
  }
}
