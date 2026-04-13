# Phase 4: Video Streaming and WebRTC - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-13
**Phase:** 04-video-streaming-and-webrtc
**Mode:** discuss
**Areas discussed:** Video streaming hook architecture, Canvas rendering, Video source selection, Connection state and error handling

---

## Video Streaming Hook Architecture

### Hook API Design

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal API | Only { isStreaming, isConnecting, error, startStream, stopStream } — keeps hook simple | |
| Rich API | Add connectionState, frameRate, sessionInfo, lastHeartbeat — gives components more visibility into stream health | ✓ |

**User's choice:** Rich API (from D-03, D-11 — expose detailed states like WebRTCService)
**Notes:** User emphasized "1:1 to match with lit so avoid extra is 1:1 migration" — mirror Lit's exact state surface

### Session Lifecycle

| Option | Description | Selected |
|--------|-------------|----------|
| One session per hook instance | Each useWebRTCStream call creates/manages its own peer connection | |
| Singleton session manager | Hook delegates to a global WebRTC session manager | ✓ |

**User's choice:** Singleton session manager
**Notes:** Reuse Lit's WebRTCService pattern — the service already manages sessions globally

### Signaling Integration

| Option | Description | Selected |
|--------|-------------|----------|
| Direct WebSocket in hook | Hook creates and manages WebSocket connections directly | |
| Reuse Lit service pattern | Create React version of WebRTCService class, hook delegates to it | ✓ |

**User's choice:** Reuse Lit service pattern
**Notes:** User emphasized "the same as we have on lit, this is a migration project"

### Filter Updates

| Option | Description | Selected |
|--------|-------------|----------|
| Via startStream parameters | startStream(filters: Filter[]) — filters passed when stream starts, must restart to change filters | ✓ |
| Dynamic filter updates | startStream() + setFilters() method — can change filters mid-stream without restarting | |

**User's choice:** Via startStream parameters
**Notes:** User said "1 once again 1:1 to match with lit" — mirror Lit's filter passing pattern

---

## Canvas Rendering Approach

### Frame Rendering

| Option | Description | Selected |
|--------|-------------|----------|
| Direct canvas manipulation | Hook draws directly to canvas ref, avoiding React state | ✓ |
| React-controlled canvas | Canvas wrapped in React component with state-managed image source | |

**User's choice:** Direct canvas manipulation
**Notes:** User said "avoid extra is 1:1 migration" — mirror Lit's CameraPreview pattern

### Buffer Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Double buffering | Alternate between two canvas elements to avoid tearing | |
| Single canvas with requestAnimationFrame | Draw each frame as it arrives, rely on browser's compositing | ✓ |

**User's choice:** Single canvas with requestAnimationFrame (matches Lit's `setInterval` pattern)

### Frame Rate Display

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, show FPS | Like Lit's stats-panel.ts displays frame metrics | ✓ |
| No, not needed | Keep UI simpler, focus on visual output only | |

**User's choice:** Yes, show FPS
**Notes:** User said "show frame rate metrics like Lit's stats-panel.ts"

---

## Video Source Selection

### Source Selection UI

| Option | Description | Selected |
|--------|-------------|----------|
| Radio buttons | "Camera" or "File" radio button toggle | |
| Dropdown | Select dropdown with camera/file options | |
| Tabs | Tab-based switching between camera and file sources | |
| Mirror Lit pattern | Use VideoSourceCard / addSource pattern | ✓ |

**User's choice:** Mirror Lit pattern
**Notes:** User said "Look to the Lit version should be the same"

### File Source Handling

| Option | Description | Selected |
|--------|-------------|----------|
| File input | Standard <input type="file"> to select video file | |
| File list integration | Reuse the FileList component from Phase 3 to select from uploaded videos | ✓ |

**User's choice:** File list integration
**Notes:** User said "reuse" — maintain consistency with Phase 3 components

### Source Switching

| Option | Description | Selected |
|--------|-------------|----------|
| Only before stream starts | Source locked once streaming begins | |
| Anytime | Can switch mid-stream, automatically restarts with new source | ✓ |

**User's choice:** Anytime
**Notes:** User said "same as Lit versionbehaivor" — sources can be added/removed dynamically

---

## Connection State and Error Handling

### Connection States

| Option | Description | Selected |
|--------|-------------|----------|
| Simple boolean states | isConnecting, isStreaming, hasError | |
| Detailed states | connecting, connected, disconnected, failed, reconnecting | ✓ |

**User's choice:** Detailed states
**Notes:** User said "1. detailed" — match WebRTCService's connectionState

### Error Toast Timing

| Option | Description | Selected |
|--------|-------------|----------|
| On any WebRTC error | Show toast immediately on any error state | |
| Only on connection failures | Show toast only when connection can't be established, not on minor issues | ✓ |

**User's choice:** Only on connection failures
**Notes:** User said "2. on connection failures" — camera errors and WebSocket failures only

### Reconnection Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-retry with backoff | Automatically attempt to reconnect with increasing delays | |
| Manual restart only | User must manually stop and start the stream again | ✓ |

**User's choice:** Manual restart only
**Notes:** User said "3. manual, look to the code for lit version it is inside the version" — Lit has no auto-retry

---

## the agent's Discretion

No areas where user said "you decide" — all decisions made explicitly based on Lit parity.

---

## Deferred Ideas

None — discussion stayed within phase scope and aligned with 1:1 migration goal.

---
*Phase: 04-video-streaming-and-webrtc*
*Context gathered: 2026-04-13*
