# Phase 4: Video Streaming and WebRTC - Technical Research

**Researched:** 2026-04-13
**Researcher:** GSD Phase Researcher
**Purpose:** Understand WebRTC integration patterns for React migration from Lit

## Research Summary

Phase 4 requires implementing real-time video streaming with WebRTC in the React frontend. This is a **Level 2 research** task (standard pattern adaptation) - we're adapting an existing Lit implementation to React, following established patterns from Phase 2 and Phase 3.

**Key insight:** The Lit implementation is complete and production-ready. The React version should mirror the architecture exactly, using React hooks for state management while preserving the WebRTC service logic.

## Canonical Reference Implementation (Lit)

### WebRTCService (`front-end/src/infrastructure/connection/webrtc-service.ts`)

**Architecture:** Singleton service class managing WebRTC peer connections and signaling

**Key capabilities:**
- `initialize()` - Browser support detection and setup
- `createPeerConnection()` - RTCPeerConnection with STUN server (stun.l.google.com:19302)
- `createDataChannel()` - RTCDataChannel for ping-pong heartbeat
- `createSession(sourceId)` - Full session lifecycle: peer connection → data channel → WebSocket signaling → ICE exchange → heartbeat
- `closeSession(sessionId)` - Cleanup: stop heartbeat, close WebSocket, close peer connection
- `getConnectionStatus()` - Returns `{ state, lastRequest, lastRequestTime }` for UI status

**Signaling protocol:**
- WebSocket endpoint: `/ws/webrtc-signaling`
- Protobuf: `SignalingMessage` with oneof messages:
  - `startSession` (StartSessionRequest: sessionId, sdpOffer)
  - `startSessionResponse` (StartSessionResponse: sessionId, sdpAnswer)
  - `iceCandidate` (IceCandidate: candidate, sdpMid, sdpMlineIndex)
  - `closeSession` (CloseSessionRequest: sessionId)
  - `keepAlive` (empty)
- Bidirectional streaming via `WebRTCSignalingService.signalingStream`

**Heartbeat mechanism:**
- Ping-pong via data channel (label: 'ping-pong-channel')
- Interval: 5000ms (5 seconds)
- Sends 'ping' on interval, expects 'pong' response
- Updates `lastRequest` and `lastRequestTime` on successful pong
- Stops on data channel close/error

**Connection states:**
- `connecting` - Initial connection setup
- `connected` - Data channel open, heartbeat active
- `disconnected` - No active sessions or heartbeat
- `error` - Connection failure

**Error handling:**
- Browser support check before initialization
- Graceful degradation if WebRTC not supported
- Automatic cleanup on peer connection failure
- Detailed logging with structured data (sessionId, connection state, ICE state)

### CameraPreview (`front-end/src/components/video/camera-preview.ts`)

**Architecture:** Lit Web Component wrapping HTML5 video and canvas elements

**Key capabilities:**
- `start()` - `navigator.mediaDevices.getUserMedia()` for camera access
- `stop()` - Stop all media tracks, clear stream, update status
- `startCapture(onFrameCallback)` - Frame capture via `setInterval` at configured FPS
- `setProcessing(isProcessing)` - Throttle capture during processing

**Frame capture pattern:**
```typescript
setInterval(() => {
  ctx.drawImage(videoElement, 0, 0, width, height);
  const dataUrl = canvas.toDataURL('image/jpeg', quality);
  const base64data = dataUrl.split(',')[1];
  onFrameCallback(base64data, width, height, timestamp);
}, 1000 / fps);
```

**Error handling:**
- `NotAllowedError` - Permission denied
- `NotFoundError` - No camera found
- `NotReadableError` - Camera in use
- Generic errors fallback to message text
- Error notifications via `ToastManager`
- Status updates via `StatsManager.updateCameraStatus(status, type)`

**Performance optimizations:**
- Video element hidden (`opacity: 0`)
- Canvas hidden (`display: none`)
- Direct canvas manipulation (no React state for frames)
- `willReadFrequently: true` for canvas context

### VideoGrid (`front-end/src/components/video/video-grid.ts`)

**Architecture:** Lit component managing multiple video sources (camera, video file, static image)

**Key capabilities:**
- `addSource(InputSource)` - Add source, create WebSocket/WebRTC session, start streaming
- `removeSource(sourceId)` - Cleanup: close session, disconnect WebSocket, stop camera
- `selectSource(sourceId)` - Dispatch `source-selection-changed` event
- `applyFilterToSelected(filters, accelerator, resolution)` - Update filters for selected source

**Source types:**
- `camera` - Uses `CameraPreview` component + `WebSocketService.sendFrame()`
- `video` - Uses `WebSocketService.sendStartVideo()` for file-based video
- Static images (default) - Uses `WebSocketService.sendSingleFrame()`

**WebRTC integration:**
```typescript
webrtcService.createSession(uniqueId).then((session) => {
  source.webrtcSession = session;
  // Heartbeat starts automatically when data channel opens
});
```

**Filter integration:**
- Reuses `FilterData` value objects from domain layer
- Maps `ActiveFilterState[]` to `FilterData[]`
- For video sources: `sendStopVideo()` → 200ms delay → `sendStartVideo()` with new filters
- For camera: Filter update not yet implemented (known limitation)

## Existing React Infrastructure

### Service Layer (Phase 2)

**ServiceContext** (`front-end/src/react/context/service-context.tsx`):
```typescript
export type GrpcClients = {
  imageProcessorClient: PromiseClient<typeof ImageProcessorService>;
  remoteManagementClient: PromiseClient<typeof RemoteManagementService>;
};
```

**useToast** (`front-end/src/react/hooks/useToast.ts`):
- Context-based toast notifications
- Methods: `error(title, message)`, `warning(title, message)`, `success(title, message)`

**useFilters** (`front-end/src/react/hooks/useFilters.ts`):
- Wraps `imageProcessorClient.listFilters()`
- Returns: `{ filters, loading, error, refetch }`
- Follows same pattern as Lit's filter list fetching

### Components (Phase 3)

**FilterPanel** (`front-end/src/react/components/filters/FilterPanel.tsx`):
- Reusable filter selection UI
- Props: `filters?`, `onFiltersChange: (ActiveFilterState[]) => void`, `initialActiveFilters?`
- Supports drag-and-drop reordering
- Parameter controls: SELECT, RANGE, NUMBER, CHECKBOX
- Exports `ActiveFilterState` interface (id, parameters)

**FileList** (`front-end/src/react/components/files/FileList.tsx`):
- Reusable file/image selection UI
- Props: `images: StaticImage[]`, `selectedImageId?`, `onImageSelect: (image) => void`, `layout?`
- Grid/list layout options
- Shows default badge for default images

## Integration Points

### WebRTC Signaling (Existing)

**WebSocket endpoint:** `/ws/webrtc-signaling`
**Protobuf types:** (`front-end/src/gen/webrtc_signal_pb.ts`)
- `StartSessionRequest` - sessionId, sdpOffer
- `StartSessionResponse` - sessionId, sdpAnswer
- `IceCandidate` - candidate, sdpMid, sdpMlineIndex
- `CloseSessionRequest` - sessionId
- `SignalingMessage` - Oneof wrapper for all message types

**Service definition:** (`front-end/src/gen/webrtc_signal_connect.ts`)
```typescript
export const WebRTCSignalingService = {
  typeName: "cuda_learning.WebRTCSignalingService",
  methods: {
    signalingStream: {
      name: "SignalingStream",
      I: SignalingMessage,
      O: SignalingMessage,
      kind: MethodKind.BiDiStreaming,
    },
  }
};
```

### Filter Selection (Phase 3)

- `FilterPanel` component provides filter UI
- `ActiveFilterState[]` passed to video streaming component
- Filters converted to `FilterData` value objects for backend

### File Source Selection (Phase 3)

- `FileList` component provides image/video selection
- `StaticImage` type from protobuf
- Default video selection via `image.isDefault` flag

## React Adaptation Strategy

### Decision: Class Service + Hook Wrapper (per D-01, D-02)

**Rationale:** Matches Phase 2 pattern where `useFilters` wraps `ListFilters` RPC

**Architecture:**
```
ReactWebRTCService (class) - Adapted from Lit WebRTCService
  └── manageWebRTC: singleton instance
  └── Methods: initialize, createSession, closeSession, getConnectionStatus, etc.

useWebRTCStream (hook) - Wraps ReactWebRTCService
  └── State: connectionStatus, activeSession, error
  └── Methods: startStream, stopStream, updateFilters
  └── Effects: cleanup on unmount
```

**File structure:**
```
front-end/src/react/infrastructure/connection/
  ├── ReactWebRTCService.ts  (class)
  └── webrtc-manage.ts       (singleton export)

front-end/src/react/hooks/
  └── useWebRTCStream.ts     (hook)
```

### Canvas Rendering Pattern

**Decision: Direct canvas manipulation (per D-05, D-06)**

**Rationale:** Matches Lit's `CameraPreview` - avoids React state for high-frequency updates

**Implementation:**
```typescript
// useRef for canvas element
const canvasRef = useRef<HTMLCanvasElement>(null);
const ctx = useRef<CanvasRenderingContext2D | null>(null);

// Initialize context once
useEffect(() => {
  if (canvasRef.current) {
    ctx.current = canvasRef.current.getContext('2d', { willReadFrequently: true });
  }
}, []);

// Frame rendering via requestAnimationFrame (not React state)
const renderFrame = useCallback((base64data: string, width: number, height: number) => {
  if (!ctx.current || !canvasRef.current) return;
  const img = new Image();
  img.onload = () => {
    ctx.current!.drawImage(img, 0, 0, width, height);
  };
  img.src = `data:image/jpeg;base64,${base64data}`;
}, []);
```

### Connection State Management

**Decision: Expose detailed states (per D-11)**

**State interface:**
```typescript
type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'failed';

interface WebRTCStreamState {
  state: ConnectionState;
  lastRequest: string | null;
  lastRequestTime: Date | null;
  activeSessionId: string | null;
  fps?: number;
  error?: Error;
}
```

**Hook returns:**
```typescript
const {
  connectionState,      // Connection status for UI
  isStreaming,         // Boolean for quick checks
  startStream,         // (sourceId, filters) => Promise<void>
  stopStream,          // () => Promise<void>
  error                // Last error for display
} = useWebRTCStream();
```

### Error Handling Pattern

**Decision: Toast notifications only on failures (per D-12)**

**Error types to notify:**
- Camera permission errors (`NotAllowedError`)
- Camera not found (`NotFoundError`)
- Camera in use (`NotReadableError`)
- WebSocket connection failures
- WebRTC peer connection failures
- Data channel errors

**Error types to handle silently:**
- Frame processing errors (logged, not notified)
- ICE candidate failures (logged, not notified)
- Heartbeat timeouts (logged, not notified)

**Implementation:**
```typescript
const { error: showError } = useToast();

try {
  await navigator.mediaDevices.getUserMedia(constraints);
} catch (err) {
  if (err.name === 'NotAllowedError') {
    showError('Permission Denied', 'Please allow camera access in your browser settings');
  } else if (err.name === 'NotFoundError') {
    showError('No Camera Found', 'No camera device detected on this system');
  }
  // ... other error types
}
```

### Resource Cleanup Pattern

**Decision: Manual restart only, full cleanup on stop (per D-13)**

**Cleanup checklist:**
1. Stop heartbeat interval
2. Send `closeSession` message via WebSocket
3. Close WebSocket connection
4. Close RTCDataChannel
5. Close RTCPeerConnection
6. Stop all media tracks (camera)
7. Clear canvas
8. Reset state to `disconnected`

**Implementation:**
```typescript
useEffect(() => {
  return () => {
    // Cleanup on unmount
    stopStream();
  };
}, []);

const stopStream = async () => {
  // Full cleanup sequence
  service.stopHeartbeat(sessionId);
  await service.closeSession(sessionId);
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
  }
  // ... other cleanup
};
```

### Filter Integration Pattern

**Decision: Restart stream to change filters (per D-15)**

**Rationale:** Matches Lit's behavior - filters must be passed when starting stream

**Implementation:**
```typescript
// Start stream with filters
const startStream = async (sourceId: string, filters: ActiveFilterState[]) => {
  const filterData = mapFiltersToValueObjects(filters);
  // ... WebRTC setup with filters
};

// Update filters requires restart
const updateFilters = async (newFilters: ActiveFilterState[]) => {
  if (isStreaming) {
    await stopStream();
    await startStream(currentSourceId, newFilters);
  }
};
```

### Video Source Selection Pattern

**Decision: Reuse FileList for file sources (per D-09)**

**Implementation:**
```typescript
// Source selection UI
<div className="source-selector">
  <div className="source-type-tabs">
    <button onClick={() => setSourceType('camera')}>Camera</button>
    <button onClick={() => setSourceType('file')}>File</button>
  </div>

  {sourceType === 'file' && (
    <FileList
      images={availableVideos}
      selectedImageId={selectedVideoId}
      onImageSelect={(video) => setSelectedVideoId(video.id)}
      layout="grid"
    />
  )}

  <button onClick={() => startStream(sourceId, selectedFilters)}>
    Start Stream
  </button>
</div>
```

## Technology Considerations

### WebRTC Browser Support

**Required APIs:**
- `RTCPeerConnection` - Core peer connection API
- `RTCDataChannel` - Data channel for heartbeat
- `navigator.mediaDevices.getUserMedia()` - Camera access
- `WebSocket` - Signaling channel

**Browser compatibility:** Modern browsers (Chrome, Firefox, Safari, Edge)

**Security requirement:** HTTPS required for `getUserMedia()` (same as Lit)

### Performance Considerations

**Frame rate:** Configurable via `fps` prop (default: 15 fps, same as Lit)

**Canvas optimization:**
- `willReadFrequently: true` for canvas context
- Direct canvas manipulation (no React state)
- `requestAnimationFrame` for smooth rendering
- Single canvas (no double buffering, per D-06)

**Memory management:**
- Cleanup on component unmount
- Cleanup on stream stop
- Clear canvas references
- Revoke blob URLs if used

### State Management Decision

**Decision: Use hooks, not Context (per Phase 2 pattern)**

**Rationale:**
- Video streaming is component-scoped (not global)
- Simpler than Context for single instance
- Matches `useFilters` pattern from Phase 2
- Easier to test and reason about

## Testing Strategy

### Unit Tests (Vitest + React Testing Library)

**Test targets:**
- `ReactWebRTCService` class methods
- `useWebRTCStream` hook behavior
- Canvas rendering logic
- Error handling paths

**Test patterns:**
```typescript
// Mock WebRTC APIs
global.RTCPeerConnection = mockRTCPeerConnection;
global.navigator.mediaDevices = mockMediaDevices;

// Test hook behavior
const { result } = renderHook(() => useWebRTCStream());
await act(async () => {
  await result.current.startStream('camera-1', []);
});
expect(result.current.isStreaming).toBe(true);
```

### Integration Tests

**Test targets:**
- Full stream lifecycle (start → receive frames → stop)
- Filter application and stream restart
- Error recovery scenarios
- Source switching

### Manual Testing

**Test scenarios:**
1. Start camera stream, verify frames render
2. Switch to file source, verify playback
3. Apply filters, verify visual changes
4. Stop stream, verify camera light turns off
5. Disconnect network, verify error toast
6. Reconnect, verify manual restart works

## Known Limitations (from Lit)

1. **Filter updates for camera not implemented** - Requires stream restart (by design per D-13)
2. **Multiple concurrent streams** - Lit supports up to 9 sources, React will implement single stream for MVP
3. **Video recording** - Not in scope (real-time streaming only)
4. **Screen sharing** - Not in scope (camera/file only)

## Implementation Phases

### Phase 4a: Core WebRTC Service (Plan 01)
- Create `ReactWebRTCService` class
- Adapt from Lit `WebRTCService`
- Implement peer connection, data channel, signaling
- Write unit tests

### Phase 4b: useWebRTCStream Hook (Plan 02)
- Create `useWebRTCStream` hook
- Wrap `ReactWebRTCService`
- Manage connection state
- Expose start/stop methods
- Write unit tests

### Phase 4c: Video Streaming Components (Plan 03)
- Create `VideoCanvas` component (canvas rendering)
- Create `VideoSourceSelector` component (source selection)
- Create `VideoStreamer` component (orchestration)
- Reuse `FilterPanel` and `FileList` from Phase 3
- Write unit tests

### Phase 4d: Integration and Testing (Plan 04)
- Wire components together in main App
- End-to-end testing
- Error handling validation
- Resource cleanup verification
- Manual testing checklist

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| WebRTC API changes | Low | Medium | Use standard WebRTC APIs, polyfill if needed |
| Browser compatibility | Low | Low | Test on Chrome, Firefox, Safari, Edge |
| Performance issues (lag) | Medium | High | Use direct canvas manipulation, avoid React state |
| Resource leaks | Medium | High | Strict cleanup on unmount/stop, add leak detection |
| Signaling protocol changes | Low | Medium | Follow existing protobuf schema, no backend changes |

## Dependencies

### External Dependencies (Already in project)
- `@connectrpc/connect` - gRPC/WebSocket client
- `@bufbuild/protobuf` - Protobuf serialization
- React 18+ - UI framework
- TypeScript 5+ - Type safety

### Internal Dependencies
- `front-end/src/gen/webrtc_signal_pb.ts` - Signaling protobuf types
- `front-end/src/gen/webrtc_signal_connect.ts` - Signaling service definition
- `front-end/src/react/context/service-context.tsx` - gRPC clients
- `front-end/src/react/hooks/useToast.ts` - Error notifications
- `front-end/src/react/components/filters/FilterPanel.tsx` - Filter selection
- `front-end/src/react/components/files/FileList.tsx` - File selection

## Success Criteria Alignment

| Requirement | Implementation Approach |
|-------------|------------------------|
| VID-01: Start real-time stream | `useWebRTCStream.startStream(sourceId, filters)` |
| VID-02: Canvas rendering at full frame rate | `VideoCanvas` component with direct canvas manipulation |
| VID-03: Select video source | `VideoSourceSelector` with camera/file tabs, reuse `FileList` |
| VID-04: Stop stream and cleanup | `useWebRTCStream.stopStream()` with full cleanup sequence |
| VID-05: Error notification on failure | `useToast.error()` in error handlers |

## Open Questions

1. **Stats panel integration** - Should React create a `StatsPanel` component similar to Lit, or use a simpler status display?
   - **Recommendation:** Create minimal status display for MVP (fps, connection state), defer full stats panel to Phase 5

2. **Frame rate metrics** - How to measure and display FPS?
   - **Recommendation:** Count frames received per second in the data channel message handler, display in status component

3. **Multiple concurrent streams** - Should React support multiple streams like Lit (up to 9)?
   - **Recommendation:** Implement single stream for MVP (simpler, meets requirements), add multi-stream support in v1.1 if needed

## Next Steps

1. **Create RESEARCH.md** (this document) ✓
2. **Create VALIDATION.md** (validation architecture for Dimension 8)
3. **Proceed to planning** - Use research findings to create 3-4 plans
4. **Execute plans** - Implement WebRTC streaming in React

---

*Research completed: 2026-04-13*
*Research type: Level 2 (Standard pattern adaptation)*
*Researcher notes: Lit implementation is well-structured and can be adapted to React with minimal changes. Focus on maintaining parity while following React patterns.*
