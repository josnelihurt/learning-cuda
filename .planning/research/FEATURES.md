# Feature Research

**Domain:** React frontend for real-time CUDA image/video processing platform
**Researched:** 2026-04-12
**Confidence:** HIGH (React patterns), MEDIUM (WebRTC-specific hooks), HIGH (gRPC/ConnectRPC integration)

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features that must exist for the React frontend to feel complete. Missing these means the migration is not done.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Filter selection panel | Core product interaction — identical to Lit `filter-panel.ts` | LOW | Stateless presentation; filter state lives in a hook or context |
| Image upload with progress | Users need feedback on upload state; existing `image-upload.ts` provides reference | MEDIUM | Use XMLHttpRequest `upload.onprogress` event; NOT fetch (fetch lacks progress events). Hook: `useImageUpload` |
| Frame-by-frame canvas display | Processed frames must render in real time; Lit `camera-preview.ts` reference | HIGH | Use `useRef<HTMLCanvasElement>` + `requestVideoFrameCallback` or `requestAnimationFrame`; never re-render via React state on each frame — that is a full React reconciliation per frame |
| Video source selector | Users select input video; Lit `video-selector.ts` + `video-grid.ts` reference | LOW | Simple stateful list; hook: `useVideoSources` wrapping ConnectRPC `ListVideos` call |
| WebRTC streaming session | Video pipeline flows through WebRTC data channel; Lit `webrtc-frame-transport-service.ts` reference | HIGH | Hook: `useWebRTCStream`; manages RTCPeerConnection lifecycle, ICE candidates, SDP offer/answer via WebSocket signaling. Cleanup on unmount is mandatory |
| Toast notification system | Error feedback mechanism; Lit `toast-container.ts` reference | LOW | Use React Context + reducer for toast queue; do NOT reach for a library when a 40-line custom implementation suffices |
| Health / connection status display | Lit `connection-status-card.ts` reference; users need backend status | LOW | Hook: `useHealthMonitor` polling `/health` endpoint on interval; clean up interval on unmount |
| System settings / configuration UI | Lit exposes config via gRPC; users need to read and modify settings | MEDIUM | Hook: `useConfiguration` wrapping ConnectRPC `GetConfiguration`/`SetConfiguration` |
| gRPC status error modal | Lit `grpc-status-modal.ts` + `grpc-unavailable.ts`; backend connectivity errors need dedicated UI | MEDIUM | Surface ConnectRPC `ConnectError` codes in a modal; error boundary does NOT catch async gRPC errors — handle in hooks |
| File management list | Uploaded images/videos must be listable; Lit `image-selector-modal.ts` reference | LOW | Hook: `useFileList` wrapping ConnectRPC `ListImages`/`ListVideos` |

### Differentiators (React-Specific Improvements)

Features that would make the React frontend meaningfully better than the Lit reference implementation.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `useWebRTCStream` hook with circuit breaker | Lit's WebRTC code is imperative and scattered; a hook isolating all RTCPeerConnection state makes reconnection logic and error handling testable in isolation | MEDIUM | Implement exponential backoff reconnect; disable auto-reconnect after N failures (circuit breaker) to avoid infinite reconnect loops |
| Error boundaries scoped to streaming zone | Lit has no component-level error isolation; a streaming crash should not tear down the filter panel or settings | MEDIUM | Place `<ErrorBoundary>` around the `<CameraPreview>` canvas zone only; use `react-error-boundary` library (standard community choice); async errors from hooks must be re-thrown into render cycle via `useState` to be caught |
| `useAsyncGRPC` hook with typed errors | Centralize ConnectRPC call lifecycle (loading / data / error) so components never manage fetch state directly | MEDIUM | Pattern: `const { data, loading, error } = useAsyncGRPC(() => client.processImage(req))`. Wrap `ConnectError` into a typed domain error before surfacing to UI |
| Stats panel with live metrics | Lit `stats-panel.ts` exists; React re-renders can be avoided by writing directly to a DOM ref for per-frame counters | LOW | Use `useRef` + direct DOM mutation for FPS counter; never put high-frequency metrics in `useState` |
| Zustand for processing session state | Filter selection, active source, and processing parameters need to survive navigation; React Context re-renders all consumers on any change — Zustand is surgically selective | MEDIUM | Scope: processing session slice only. Auth/config can remain in Context. Zustand requires no provider boilerplate |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| `useState` for per-frame video data | Seems natural — just store the latest frame in state | Every `setState` call triggers React reconciliation. At 30fps that is 30 reconciliations/sec. Frame display will lag, drop frames, or cause jank | Store frame data in `useRef`; draw to canvas imperatively in `requestAnimationFrame` callback. Keep React state out of the render loop |
| Redux for all state | Teams default to Redux; it is familiar | Extreme overhead for a single-user local app with no server state synchronization requirement. Redux DevTools are useful but the boilerplate and bundle cost are not justified here | React Context for global UI state (toast, health); Zustand for processing session state; local `useState` for component-local concerns |
| Real-time UI updates via WebSocket for every processing metric | Metrics feel important to stream live | WebSocket messages triggering `setState` on every frame floods the React tree. Same jank problem as frame state above | Batch metric updates with `useEffect` on a 500ms interval; write counters to refs for display; only call `setState` for status transitions (connected/disconnected/error) |
| `useEffect` for gRPC streaming subscription setup | Seems correct — effects handle subscriptions | `useEffect` runs after every render by default and dependency arrays are easy to misspecify, causing duplicate stream connections or stale closures | Use `useRef` to track whether stream is already established; use an `AbortController` for cleanup. For complex streaming, a dedicated service class initialized once in Context is safer than effect-based setup |
| Lifting all WebRTC state to app root | Avoids prop drilling | RTCPeerConnection is a heavyweight browser resource. Putting it at root means it lives forever and its error state pollutes the entire app tree | Instantiate WebRTC inside a `useWebRTCStream` hook scoped to the video player component; the hook cleans up the peer connection on unmount |
| Per-component gRPC client instantiation | Seems modular | Creates multiple transport connections, breaks request batching, inflates bundle initialization cost | Create one transport instance at app boot; pass client instances via React Context or import from a singleton module |

---

## Feature Dependencies

```
[WebRTC Streaming Session]
    └──requires──> [WebSocket signaling channel]
                       └──requires──> [Go server WebSocket handler] (already exists)
    └──requires──> [Video source selector] (user must pick source first)

[Frame Canvas Display]
    └──requires──> [WebRTC Streaming Session] OR [gRPC image processing response]

[Filter Selection Panel]
    └──requires──> [useFilters hook] (fetches available filters via ConnectRPC GetCapabilities)

[Image Upload]
    └──enhances──> [Frame Canvas Display] (uploaded image can be processed and displayed)

[Error Boundaries]
    └──wraps──> [WebRTC Streaming Session component zone]
    └──wraps──> [Image Processing component zone]

[useAsyncGRPC hook]
    └──required by──> [Filter fetch]
    └──required by──> [Image upload processing]
    └──required by──> [File management list]
    └──required by──> [Configuration UI]
    └──required by──> [Health monitor]

[ConnectRPC client singleton]
    └──required by──> [useAsyncGRPC hook]
    └──required by──> [useWebRTCStream] (for WebRTC signaling bootstrap)
```

### Dependency Notes

- **WebRTC requires WebSocket signaling first:** The Go server already implements WebSocket SDP signaling at `pkg/interfaces/websocket/`. The React hook must open a WebSocket connection before attempting RTCPeerConnection offer/answer.
- **Frame display must bypass React state:** The canvas frame rendering path must not flow through React state to avoid reconciliation overhead. This is a hard architectural constraint, not a preference.
- **ConnectRPC client must be a singleton:** The existing `webserver/web/src/gen/` protobuf-generated clients and transport should be initialized once and shared via Context or module-level singleton. The Lit frontend does this via its DI container; React replaces the DI container with Context.
- **Error boundaries cannot catch async errors directly:** gRPC call failures and WebRTC ICE failures are async. They must be promoted into render-cycle errors via `setState` to be caught by an error boundary. The `react-error-boundary` library's `useErrorBoundary` hook provides the standard mechanism for this.

---

## MVP Definition

This is a migration, not a greenfield product. MVP means full feature parity with the Lit frontend.

### Launch With (v1 — Phase 1-3)

- [ ] React + Vite + TypeScript scaffold with `/react` route served by Go static handler — enables parallel dev
- [ ] ConnectRPC client singleton initialized in React app bootstrap — all subsequent hooks depend on this
- [ ] `useFilters` hook + Filter selection panel — core product interaction
- [ ] `useImageUpload` hook + image upload UI with progress — first complete user workflow
- [ ] `useAsyncGRPC` hook — shared infrastructure for all gRPC calls
- [ ] Toast notification context + `useToast` hook — error feedback for all hooks

### Add After Scaffold Is Working (v1.x — Phase 4-5)

- [ ] `useWebRTCStream` hook — most complex hook, depends on solid hook patterns established in earlier phases
- [ ] Canvas frame display component — requires WebRTC hook
- [ ] Video source selector and management
- [ ] Health monitor + connection status display
- [ ] Error boundaries scoped to streaming zone — added after streaming is working

### Final Phase (v1 completion)

- [ ] Configuration / settings UI
- [ ] Stats panel (FPS, latency display via refs)
- [ ] Full React Testing Library test coverage
- [ ] Dual-route comparison validation (Lit vs React parity check)
- [ ] CSS reuse audit and cleanup

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| ConnectRPC client singleton + useAsyncGRPC | HIGH | LOW | P1 |
| Filter selection panel | HIGH | LOW | P1 |
| Toast notification system | HIGH | LOW | P1 |
| Image upload with progress | HIGH | MEDIUM | P1 |
| WebRTC streaming session (useWebRTCStream) | HIGH | HIGH | P1 |
| Canvas frame display (ref-based) | HIGH | MEDIUM | P1 |
| Video source selector | MEDIUM | LOW | P1 |
| Health monitor | MEDIUM | LOW | P2 |
| Error boundaries (streaming zone) | HIGH | MEDIUM | P2 |
| useAsyncGRPC with typed errors | MEDIUM | MEDIUM | P2 |
| Configuration UI | MEDIUM | MEDIUM | P2 |
| File management list | MEDIUM | LOW | P2 |
| Stats panel (ref-based counters) | LOW | LOW | P3 |
| Zustand for session state | MEDIUM | LOW | P3 — only if Context causes measurable re-render problems |

**Priority key:**
- P1: Required for feature parity — migration is incomplete without this
- P2: Required for production quality — should not ship without this
- P3: Improvement over Lit — defer until parity is confirmed

---

## React-Specific Pattern Notes

These are React patterns specific to this domain, not generic web patterns.

### WebRTC Hook Structure

The `useWebRTCStream` hook must manage the full RTCPeerConnection lifecycle:

```
state machine: idle → connecting → connected → failed → reconnecting → idle
```

The hook owns: RTCPeerConnection, WebSocket signaling channel, ICE candidate queuing, SDP offer/answer, and cleanup. Components receive only: `{ streamState, startStream, stopStream, lastError }`. Components never touch the RTCPeerConnection directly.

Cleanup must call `peerConnection.close()` AND close the WebSocket signaling channel on unmount. Failing to close the peer connection leaks TURN/STUN resources and leaves dangling ICE timers.

### Canvas Frame Display — No React State in Render Loop

```
Wrong: frame arrives → setState(frame) → React re-renders → canvas draws
Right: frame arrives → ref.current.drawImage(frame) [direct imperative call]
```

Use `requestVideoFrameCallback` (Chrome/Edge) with `requestAnimationFrame` fallback for the draw loop. The `useRef<HTMLCanvasElement>` ref is the only React primitive needed; all else is imperative canvas API.

### gRPC Error Handling

`ConnectError` from `@connectrpc/connect` carries a `code` field (ConnectRPC status code) and `message`. Hooks must wrap this into a domain-typed error before surfacing to components:

```typescript
type ProcessingError =
  | { kind: 'unavailable'; message: string }
  | { kind: 'invalid_request'; message: string }
  | { kind: 'unknown'; message: string }
```

Components display errors; hooks translate error codes. Never expose `ConnectError` directly to component props.

### Error Boundary Scope

Place error boundaries at three levels:
1. App root — prevents blank screen on catastrophic failure
2. Streaming zone (`<CameraPreview>`) — streaming crash does not affect filter panel
3. Filter panel — filter fetch failure does not affect streaming

Use `react-error-boundary` (community standard, maintained, function-component compatible) rather than writing class-based boundaries manually.

---

## Sources

- [WebRTC React comprehensive guide — DEV Community](https://dev.to/bhavyajain/webrtc-react-a-comprehensive-guide-2hdk)
- [React WebRTC Video Call 2025 — VideoSDK](https://www.videosdk.live/developer-hub/webrtc/react-webrtc-video-call)
- [WebRTC on React Hooks and TypeScript — GitHub Gist](https://gist.github.com/keshihoriuchi/40ff3217a7a63d25788ce5cb8230ba3b)
- [File Upload Hook with Preview 2026 — react.wiki](https://react.wiki/hooks/file-upload-hook/)
- [File Upload Management with Progress Tracking — Medium](https://medium.com/@didemsahin1789/file-upload-management-robust-upload-system-with-progress-tracking-c5971c48f074)
- [Production-Ready Error Boundaries in React — DEV Community](https://dev.to/whoffagents/production-ready-error-boundaries-in-react-patterns-for-graceful-failures-2dan)
- [Error Handling with react-error-boundary — Certificates.dev](https://certificates.dev/blog/error-handling-in-react-with-react-error-boundary)
- [Using gRPC in React the Modern Way — DEV Community](https://dev.to/arichy/using-grpc-in-react-the-modern-way-from-grpc-web-to-connect-41lc)
- [Real-Time Data with gRPC Streaming and ConnectRPC — DEV Community](https://dev.to/dmo2000/real-time-data-with-grpc-streaming-net-react-with-connect-rpc-20i8)
- [Performant animations with requestAnimationFrame and React hooks — Medium](https://layonez.medium.com/performant-animations-with-requestanimationframe-and-react-hooks-99a32c5c9fbf)
- [Perform efficient per-video-frame operations — web.dev](https://web.dev/articles/requestvideoframecallback-rvfc)
- [React Architecture Patterns 2025 — LaunchDarkly](https://launchdarkly.com/docs/blog/react-architecture-2025)
- [gRPC Client Toolkit with React Hooks — DEV Community](https://dev.to/23n6/grpc-client-toolkit-with-interceptors-and-react-hooks-1oef)
- Existing Lit frontend codebase: `webserver/web/src/components/`, `webserver/web/src/services/`

---

*Feature research for: React frontend migration of CUDA image/video processing platform*
*Researched: 2026-04-12*
