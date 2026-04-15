# Step 5: Go Video Pipeline via WebRTC

## Goal
Add a Go-managed WebRTC video playback path that accepts a browser session ID, establishes a Go-to-C++ DataChannel, and streams serialized `ProcessImageRequest` frames from FFmpeg playback.

## Progress Table
| Task | Status | Notes |
|---|---|---|
| Resolve Step 5 contract drift | [x] | The repo only exposed `StreamProcessVideo`, but it lacked `video_id`; added dedicated `StartVideoPlayback` / `StopVideoPlayback` RPCs after confirming the gap. |
| Add playback RPC contract and regenerate clients | [x] | Updated `image_processor_service.proto`, regenerated Go/TS outputs, and exposed new Connect procedures. |
| Implement Go WebRTC peer wrapper | [x] | Added `pkg/infrastructure/webrtc/go_peer.go` using `pion/webrtc/v4` and the existing gRPC `WebRTCSignalingService`. |
| Implement `StreamVideoUseCase` + tests | [x] | Added async session management keyed by browser `session_id`, frame marshaling, stop handling, and application-layer tests. |
| Wire handler, container, app, and signaling registration | [x] | ImageProcessor handler exposes Start/Stop methods, DI builds the use case, app passes it through, and the existing signaling service is now registered. |
| Run Step 5 verification | [x] | `go build`, `go vet`, and `go test ./pkg/application/...` passed. |

## Files Touched
- `proto/image_processor_service.proto` — read, edited
- `proto/gen/image_processor_service.pb.go` — regenerated
- `proto/gen/genconnect/image_processor_service.connect.go` — regenerated
- `src/front-end/src/gen/image_processor_service_pb.ts` — regenerated
- `src/front-end/src/gen/image_processor_service_connect.ts` — regenerated
- `go.mod` — edited by `go get` / `go mod tidy`
- `go.sum` — updated by `go get` / `go mod tidy`
- `src/go_api/pkg/application/stream_video_use_case.go` — created
- `src/go_api/pkg/application/stream_video_use_case_test.go` — created
- `src/go_api/pkg/infrastructure/webrtc/go_peer.go` — created
- `src/go_api/pkg/interfaces/connectrpc/handler.go` — read, edited
- `src/go_api/pkg/interfaces/connectrpc/handler_test.go` — read, edited
- `src/go_api/pkg/container/container.go` — read, edited
- `src/go_api/pkg/app/app.go` — read, edited
- `src/go_api/cmd/server/main.go` — read, edited

## Assumptions & Drift
- Agent C's prompt expected `StartVideoPlayback` / `StopVideoPlayback`, but the repo state after Step 4 only exposed `StreamProcessVideo`.
- The authoritative source plan also referenced a new video-start RPC, and `ProcessImageRequest` had no `video_id`, so the current contract could not identify which video file Go should play. Based on that drift and explicit user confirmation, dedicated Start/Stop playback RPCs were added in Step 5.
- `pion/webrtc/v4` version `v4.2.11` was chosen because `go get` resolved it cleanly against the existing Go 1.24 module and the API surface matched the needed `PeerConnection` / `DataChannel` primitives.

## Verification Log
### Local checks
- `./scripts/build/protos.sh` → PASS (`Proto files generated successfully`)
- `cd /home/jrb/code/cuda-learning && go get github.com/pion/webrtc/v4@latest` → PASS (`github.com/pion/webrtc/v4 v4.2.11`)
- `cd /home/jrb/code/cuda-learning && go mod tidy` → PASS
- `cd /home/jrb/code/cuda-learning/src/go_api && go build ./...` → PASS
- `cd /home/jrb/code/cuda-learning/src/go_api && go vet ./...` → PASS
- `cd /home/jrb/code/cuda-learning/src/go_api && go test ./pkg/application/...` → PASS
- `rg -n 'StreamVideoUseCase' src/go_api/pkg/app src/go_api/pkg/container src/go_api/pkg/interfaces/connectrpc` → PASS

### Global checks (if run)
- Not run in this step context.

## Integration Notes
- **Signaling reuse:** `GoPeer.Connect()` opens the existing gRPC `WebRTCSignalingService.SignalingStream`, sends a `StartSessionRequest` carrying Pion's SDP offer, forwards local ICE candidates as `SendIceCandidateRequest`, applies the returned `StartSessionResponse` SDP answer, and adds remote ICE candidates from the same signaling stream.
- **DataChannel label convention:** `go-video-<browser_session_id>`. The same prefixed value is used as the Go peer's signaling `session_id`, while each marshaled `ProcessImageRequest.session_id` keeps the original browser session ID so Agent D can route processed output back to the browser peer.
- **Test coverage summary:** application tests cover successful frame marshaling/sending, stop/cancellation behavior, and repository lookup failure.

## Open Questions / Escalations
- `StreamProcessVideo` remains explicitly unimplemented in the Connect handler because the chosen Step 5 contract uses dedicated Start/Stop playback RPCs. If later agents want to converge back onto a bidi stream API, the contract will need a separate follow-up.

## Attempt Log
### Attempt 1 — 2026-04-14T23:16:31.143Z
- Reviewed Step 5 plan inputs and found contract drift: the repo had signaling and video playback infrastructure seams, but no request shape that could carry a `video_id` into Go.
- Confirmed the source plan and user decision favored dedicated Start/Stop playback RPCs, then added them to the proto, regenerated clients, and implemented the Go peer/use-case/handler wiring.
- Registered the existing WebRTC signaling Connect service in the app and passed Go build/vet/application tests.
