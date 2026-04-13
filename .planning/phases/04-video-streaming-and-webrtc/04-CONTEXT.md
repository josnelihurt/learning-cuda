# Phase 4: Video Streaming and WebRTC - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Users can start, watch, and stop a real-time filtered video stream in the React frontend with full WebRTC resource cleanup. This is a 1:1 migration from the Lit frontend — React implementation must mirror Lit's behavior exactly.

**Out of scope:** New video features, backend WebRTC changes, different streaming protocols.

</domain>

<decisions>
## Implementation Decisions

### Video Streaming Hook Architecture

- **D-01:** Mirror Lit's `WebRTCService` class pattern — create React version in `front-end/src/react/infrastructure/connection/` with identical functionality
- **D-02:** `useWebRTCStream` hook wraps the service class (same pattern as `useFilters` wraps `ListFilters` RPC from Phase 2)
- **D-03:** Expose same state surface as Lit: `getConnectionStatus()` returns `{ state, lastRequest, lastRequestTime }`, plus streaming-specific state
- **D-04:** Use same WebSocket signaling protocol to `/ws/webrtc-signaling` with `SignalingMessage` protobuf

### Canvas Rendering

- **D-05:** Direct canvas manipulation, avoiding React state — matches Lit's `CameraPreview` pattern where frames are drawn directly to canvas
- **D-06:** Same buffer handling approach as Lit — single canvas with `requestAnimationFrame` equivalent, no double buffering
- **D-07:** Show frame rate metrics like Lit's `stats-panel.ts` — display FPS and other stats for parity

### Video Source Selection

- **D-08:** Source selection UI — mirror Lit's `VideoSourceCard` pattern with `addSource(InputSource)` and type-based rendering
- **D-09:** File source handling — reuse `FileList` component from Phase 3 to select from uploaded videos (consistent with reuse pattern, file sources are `StaticVideo` objects)
- **D-10:** Source switching — allowed anytime (sources can be added/removed dynamically like Lit's `video-grid.ts`)

### Connection State and Error Handling

- **D-11:** Expose detailed connection states: `connecting`, `connected`, `disconnected`, `failed` (from `WebRTCService` pattern and `connectionState` property)
- **D-12:** Show toast notifications on connection failures only — camera permission errors, WebSocket connection failures (not minor issues)
- **D-13:** Manual restart only — no auto-retry or auto-reconnection (matches Lit behavior, user must restart stream)
- **D-14:** Use `statsManager.updateCameraStatus()` pattern for status updates with types: 'success', 'error', 'warning', 'inactive'

### Filter Integration

- **D-15:** Filters passed via `startStream(filters: Filter[])` — must restart stream to change filters (matches Lit's frame sending pattern)
- **D-16:** Reuse `FilterPanel` component from Phase 3 for filter selection and configuration

### the agent's Discretion

- Exact component file structure within `front-end/src/react/infrastructure/` and `front-end/src/react/components/video/`
- Whether to create a separate `WebRTCService` React class or adapt the existing Lit service
- Canvas sizing and resolution handling details
- Exact error message wording for toast notifications
- Stats panel integration approach for React

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Lit WebRTC Reference Implementation

- `front-end/src/infrastructure/connection/webrtc-service.ts` — Complete WebRTC service with peer connections, data channels, WebSocket signaling, heartbeat, session lifecycle (the reference implementation to mirror)
- `front-end/src/infrastructure/transport/webrtc-frame-transport.ts` — Frame transport interface (currently stubbed, needs implementation in React)
- `front-end/src/components/video/camera-preview.ts` — Camera capture, canvas rendering, frame capture with `setInterval`, error handling pattern
- `front-end/src/components/video/video-grid.ts` — Source management, `addSource()` pattern, source type handling ('camera' vs 'static')
- `front-end/src/components/video/video-source-card.ts` — Source card UI with selection state, close button, event dispatching
- `front-end/src/components/video/video-selector.ts` — File source selection using `videoService.listAvailableVideos()`, default video selection

### Protobuf and Signaling

- `front-end/src/gen/webrtc_signal_connect.ts` — `WebRTCSignalingService` with `SignalingStream` BiDiStreaming
- `front-end/src/gen/webrtc_signal_pb.js` — `SignalingMessage` with startSession, startSessionResponse, iceCandidate, closeSession message types

### Requirements and Roadmap

- `.planning/REQUIREMENTS.md` §VID-01 through VID-05 — Video streaming requirements
- `.planning/ROADMAP.md` §Phase 4 — Phase goal, success criteria, UI hint

### Prior Phase Context

- `.planning/phases/03-static-feature-ui/03-CONTEXT.md` — FilterPanel and FileList components to reuse
- `.planning/phases/02-core-hook-infrastructure/02-CONTEXT.md` — Toast notifications via `useToast`, Connect transport patterns
- `.planning/phases/01-scaffold-and-infrastructure/01-CONTEXT.md` — WebRTC stubs in test-setup.ts

### Project Constraints

- `.planning/PROJECT.md` — React learning goals, feature parity requirement, zero backend changes constraint

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- **WebRTCService** (`front-end/src/infrastructure/connection/webrtc-service.ts`) — Complete implementation with peer connection lifecycle, WebSocket signaling, heartbeat, session management. Can be adapted for React or used as reference.
- **CameraPreview** (`front-end/src/components/video/camera-preview.ts`) — Shows camera capture pattern, canvas drawing, frame capture interval, error handling with specific error types (NotAllowedError, NotFoundError, NotReadableError).
- **VideoGrid** (`front-end/src/components/video/video-grid.ts`) — Shows source management pattern, `addSource(InputSource)`, dynamic source addition/removal.
- **VideoSelector** (`front-end/src/components/video/video-selector.ts`) — Shows file selection pattern using `videoService.listAvailableVideos()`, default video handling.
- **FilterPanel** (Phase 3, `front-end/src/react/components/filters/FilterPanel.tsx`) — Reusable for filter selection in video streaming.
- **FileList** (Phase 3, `front-end/src/react/components/files/FileList.tsx`) — Reusable for file source selection.
- **useToast** (Phase 2, `front-end/src/react/hooks/useToast.ts`) — For error notifications.

### Established Patterns

- **WebRTC Signaling**: WebSocket to `/ws/webrtc-signaling`, `SignalingMessage` protobuf with `startSession`, `iceCandidate`, `closeSession` message types
- **Peer Connection Lifecycle**: `createPeerConnection()`, `createDataChannel()`, `setLocalDescription()`, `setRemoteDescription()`, ICE candidate exchange
- **Session Management**: `createSession(sourceId)`, `closeSession(sessionId)`, heartbeat with ping/pong, cleanup on error/disconnect
- **Frame Capture**: `setInterval` at configured FPS, draw video to canvas, convert to base64, callback with frame data
- **Error Handling**: Specific error types for camera (NotAllowedError, NotFoundError, NotReadableError), toast notifications, status updates via stats manager
- **State Management**: Custom hooks wrap service classes (like `useFilters`), expose minimal state surface
- **CSS Modules**: Reuse CSS custom properties from Lit for theme consistency

### Integration Points

- **WebSocket signaling endpoint**: `/ws/webrtc-signaling` (same as Lit)
- **Video source service**: `videoService.listAvailableVideos()` for file sources
- **Stats panel**: For displaying frame rate and connection status (mirror Lit's stats-panel pattern)
- **Toast notifications**: Via `useToast()` hook for error display
- **Filter integration**: Via `FilterPanel` component for selecting filters before stream starts

</code_context>

<specifics>
## Specific Ideas

- 1:1 migration — React implementation must match Lit behavior exactly, no "improvements" unless explicitly requested
- Mirror `WebRTCService` class structure for consistency
- Reuse existing filter and file selection components from Phase 3
- Camera indicator light should turn off when stream stops (VID-04 success criterion)
- Frame rate should match source frame rate without UI lag (VID-02 success criterion)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope and aligned with 1:1 migration goal.

</deferred>

---
*Phase: 04-video-streaming-and-webrtc*
*Context gathered: 2026-04-13*
