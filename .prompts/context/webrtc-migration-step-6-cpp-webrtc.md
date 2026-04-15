# Step 6: C++ WebRTC Frame Processing

## Goal
Turn the C++ WebRTC DataChannel path into a real `ProcessImageRequest` / `ProcessImageResponse` pipeline and add browser-session routing so Go-streamed video frames can be returned to the correct browser peer.

## Progress Table
| Task | Status | Notes |
|---|---|---|
| Inject `ProcessorEngine` into `WebRTCManager` | [x] | `server_main.cpp` now constructs `WebRTCManager(engine.get())`, and the manager processes incoming DataChannel frames directly through `ProcessorEngine`. |
| Replace `ping/pong` stub with proto frame processing | [x] | `onMessage` now accepts binary payloads only, parses `ProcessImageRequest`, copies response metadata, runs `ProcessImage`, serializes `ProcessImageResponse`, and sends it back over the same DataChannel. |
| Add browser-session routing via `SendToSession` | [x] | `WebRTCManager` now keeps a mutex-protected browser session → DataChannel map and `ImageProcessorServiceImpl::StreamProcessVideo` routes serialized responses when `request.session_id()` is present. |
| Distinguish browser peers from Go video peers | [x] | Adopted Agent C's `go-video-<browser_session_id>` convention and only register routable channels for non-`go-video-` sessions. |
| Extend C++ coverage for routed stream responses | [x] | Added a gRPC service test that asserts `StreamProcessVideo` calls `SendToSession` with the browser session and serialized response payload. |
| Run Step 6 verification | [x] | `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server` and `bazel test //src/cpp_accelerator/...` passed. |

## Files Touched
- `src/cpp_accelerator/ports/grpc/webrtc_manager.h` — edited
- `src/cpp_accelerator/ports/grpc/webrtc_manager.cpp` — edited
- `src/cpp_accelerator/ports/grpc/image_processor_service_impl.h` — edited
- `src/cpp_accelerator/ports/grpc/image_processor_service_impl.cpp` — edited
- `src/cpp_accelerator/ports/grpc/image_processor_service_impl_test.cpp` — edited
- `src/cpp_accelerator/ports/grpc/server_main.cpp` — edited
- `src/cpp_accelerator/ports/grpc/BUILD` — edited

## Verification Log
### Local checks
- `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server` → PASS
- `bazel test //src/cpp_accelerator/...` → PASS
- `rg -n 'ping|pong' src/cpp_accelerator/ports/grpc/webrtc_manager.cpp` → PASS (no matches)
- `rg -n 'SendToSession' src/cpp_accelerator/ports/grpc` → PASS

### Global checks (if run)
- Not run in this step context.

## Integration Notes
- **DataChannel/session convention:** Agent C's Go peer uses `go-video-<browser_session_id>` as both signaling `session_id` and DataChannel label. Step 6 treats those sessions as sender-only and only registers routable browser channels for non-`go-video-` sessions.
- **Routing ownership:** `WebRTCManager` owns a mutex-protected `session_channels_` map of browser session IDs to open DataChannels. `onOpen` registers browser channels, `onClosed` and session teardown remove them, and `SendToSession` sends serialized `ProcessImageResponse` bytes back to the browser peer.
- **Service routing seam:** `ImageProcessorServiceImpl::StreamProcessVideo` still writes each response to the gRPC stream for compatibility, but now also serializes and forwards the response to `SendToSession(request.session_id(), bytes)` when the request names a browser session.

## Thread Safety Review
- `session_channels_` is protected by its own mutex, while the existing `sessions_` and per-session mutexes continue to guard WebRTC session state.
- `ProcessorEngine::ProcessImage` does not expose shared mutable state at the `ProcessorEngine` object level after initialization; it delegates each call into a request-local `FilterPipeline`. No Step 6 escalation was needed from the current code inspection.

## Open Questions / Escalations
- None for Step 6. The next step is frontend alignment so browser peers consistently establish the routable non-`go-video-` session that `SendToSession` targets.

## Attempt Log
### Attempt 1 — 2026-04-14T23:33:16.622Z
- Read Agent D's scope, the Step 5 handoff, and the current C++ signaling/service code to confirm the Go peer convention and identify the remaining `ping/pong` stub.
- Injected `ProcessorEngine` into `WebRTCManager`, replaced text heartbeat handling with binary proto frame processing, added mutex-protected browser-session routing, and wired `StreamProcessVideo` to forward session-tagged responses back to WebRTC.
- Extended the gRPC service test coverage for routed responses and passed the Step 6 Bazel build/test gates.
