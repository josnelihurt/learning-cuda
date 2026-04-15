# Step 4: Proto & Contracts

## Goal
Add `session_id` to `ProcessImageRequest`, remove WebSocket-only proto messages, and regenerate protobuf outputs cleanly.

## Progress Table
| Task | Status | Notes |
|---|---|---|
| Read Step 4 inputs and inspect `image_processor_service.proto` | [x] | Confirmed Step 4 matched repo state. |
| Reserve field number for `ProcessImageRequest.session_id` | [x] | Field `29` was free and used. |
| Map stale callers and decide `VideoFrameUpdate` keep/delete | [x] | No non-generated `VideoFrameUpdate` callers under `src/`; message deleted. |
| Edit proto and regenerate outputs | [x] | `./scripts/build/protos.sh` passed. |
| Run C++ build gate | [x] | `bazel build //src/cpp_accelerator/...` passed. |

## Files Touched
- `proto/image_processor_service.proto` — read, edited
- `proto/gen/image_processor_service.pb.go` — regenerated
- `src/front-end/src/gen/image_processor_service_pb.ts` — regenerated
- `src/front-end/src/gen/image_processor_service_connect.ts` — regenerated (no content diff after regen)

## Assumptions & Drift
- No Step 4 drift found in the proto file.
- `ProcessImageRequest` field number `29` was available, so the planned field number was used without adjustment.
- `rg -n "\bVideoFrameUpdate\b" src --glob '!**/gen/**'` returned zero matches, so `VideoFrameUpdate` was deleted with the WebSocket-only messages.

## Verification Log
### Local checks
- `rg -n 'WebSocketFrameRequest|WebSocketFrameResponse|StartVideoPlaybackRequest|StopVideoPlaybackRequest' proto/image_processor_service.proto` → PASS (no matches)
- `rg -n 'session_id' proto/image_processor_service.proto` → PASS (`72:  string session_id = 29 [json_name = "session_id"];`)
- `./scripts/build/protos.sh` → PASS (`Proto files generated successfully`)
- `stat -c '%y %n' proto/gen/image_processor_service.pb.go src/front-end/src/gen/image_processor_service_pb.ts src/front-end/src/gen/image_processor_service_connect.ts` → PASS
  - `2026-04-14 16:02:26.373184224 -0700 proto/gen/image_processor_service.pb.go`
  - `2026-04-14 16:02:26.373505590 -0700 src/front-end/src/gen/image_processor_service_pb.ts`
  - `2026-04-14 16:02:26.373505590 -0700 src/front-end/src/gen/image_processor_service_connect.ts`

### Global checks (if run)
- `bazel build //src/cpp_accelerator/...` → PASS (`INFO: Build completed successfully, 4833 total actions`)

## Open Questions / Escalations
- None for Step 4.

## Caller Impact List
These files still reference deleted proto contracts and need follow-up updates by downstream agents:
- `src/front-end/src/domain/interfaces/IFrameTransportService.ts`
- `src/front-end/src/infrastructure/transport/frame-transport-service.ts`
- `src/front-end/src/infrastructure/transport/grpc-frame-transport.ts`
- `src/front-end/src/infrastructure/transport/webrtc-frame-transport.ts`
- `src/front-end/src/infrastructure/transport/websocket-frame-transport.ts`
- `src/front-end/src/react/infrastructure/transport/FrameTransportService.test.tsx`
- `src/front-end/src/react/infrastructure/transport/FrameTransportService.ts`
- `src/front-end/src/services/frame-transport-service.ts`
- `src/front-end/src/services/grpc-frame-transport-service.ts`
- `src/front-end/src/services/webrtc-frame-transport-service.ts`

## Attempt Log
### Attempt 1 — 2026-04-14T23:01:29.950Z
- Read the shared migration prompt, Agent B scope, source plan, and current proto.
- Confirmed field `29` was free on `ProcessImageRequest`.
- Searched `src/` callers for `WebSocketFrameRequest`, `WebSocketFrameResponse`, `StartVideoPlaybackRequest`, `StopVideoPlaybackRequest`, and `VideoFrameUpdate`.
- Deleted the WebSocket-only messages plus `VideoFrameUpdate`, added `session_id`, regenerated protobufs, and passed the C++ build gate.
