# Agent C — Go Video Pipeline via WebRTC (Step 5)

> Read `.prompts/plans/webrtc-migration-subagent-prompt.md` first. This file is YOUR scope only.

## Role
Senior Go engineer building a new use case that drives FFmpeg video frames into a `pion/webrtc` DataChannel connected to the C++ accelerator, reusing the existing gRPC `WebRTCSignalingService` for SDP/ICE exchange.

## Goal
Replace the deleted WebSocket video path with a Go-side WebRTC peer that: (1) accepts a `StartVideoPlayback` ConnectRPC request from the frontend carrying `video_id` + browser's `session_id`, (2) establishes its own DataChannel to C++, (3) streams serialized `ProcessImageRequest` frames (with `target_session_id` = browser session) so C++ can route responses to the browser's DataChannel.

## Source plan section
`~/.claude/plans/lexical-tumbling-fern.md` — Step **5**.

## Dependencies (hard gates)
- **Agent A** must have deleted the old `websocket` package and made `app.go` / `container.go` compile cleanly.
- **Agent B** must have added `session_id` to `ProcessImageRequest` and regenerated Go protos.
- Do **not** start until both show `[x]` in the INDEX.

## In-scope files
### Create
- `src/go_api/pkg/application/stream_video_use_case.go` — orchestrates FFmpeg player + WebRTC DataChannel lifecycle
- `src/go_api/pkg/application/stream_video_use_case_test.go` — AAA, `sut`, table-driven, `testify/mock` per CLAUDE.md
- `src/go_api/pkg/infrastructure/webrtc/go_peer.go` — thin `pion/webrtc` wrapper; uses existing `WebRTCSignalingService` gRPC stream for signaling

### Edit
- `src/go_api/pkg/interfaces/connectrpc/handler.go` — add `StartVideoPlayback` and `StopVideoPlayback` RPC methods wired to `StreamVideoUseCase`
- `src/go_api/pkg/container/container.go` — wire the new use case and `GoPeer` factory
- `src/go_api/pkg/app/app.go` — add option/field for `StreamVideoUseCase`, pass to ConnectRPC setup
- `go.mod` / `go.sum` — add `github.com/pion/webrtc/v4` (or current major) via `go get`

### Reuse (read-only)
- Existing `video.NewFFmpegVideoPlayer()` (was consumed by the deleted WS handler — locate via `grep`)
- Existing `WebRTCSignalingService` gRPC client/stream in the C++ server

## Out of scope
- C++ side (Agent D owns `WebRTCManager` and `image_processor_service_impl.cpp`)
- Frontend transport (Agent E)
- Proto edits (Agent B)
- WebSocket deletion (Agent A)

## Flow to implement
1. Frontend calls `StartVideoPlayback(video_id, session_id, filters, accelerator)` via ConnectRPC.
2. Handler invokes `StreamVideoUseCase.Start(ctx, req)`.
3. Use case asks `GoPeer` to open a DataChannel to C++ using `WebRTCSignalingService` for SDP/ICE. The DataChannel should be labeled distinctly (e.g. `go-video-<session_id>`) so C++ can identify Go as the sender.
4. Use case starts `FFmpegVideoPlayer` for `video_id`. Each decoded frame → build a `ProcessImageRequest` proto with `session_id = <browser session>`, filters, accelerator → `proto.Marshal` → `dataChannel.Send(bytes)`.
5. On `StopVideoPlayback` or ctx cancellation, close the player and the DataChannel.
6. Return errors through ConnectRPC; no silent failures.

## Ordering
1. `go get github.com/pion/webrtc/v4` (confirm major version with `context7` if unsure of API).
2. Implement `go_peer.go` with a minimal surface: `Connect(ctx) error`, `DataChannel() *webrtc.DataChannel`, `Close() error`.
3. Implement `stream_video_use_case.go` using `GoPeer` + existing `FFmpegVideoPlayer`.
4. Add unit tests (mock `GoPeer` and `VideoPlayer`).
5. Wire handler + container + app.
6. `cd src/go_api && go build ./... && go vet ./... && go test ./pkg/application/...`.

## Verification gates
- `go build ./...` PASS
- `go test ./pkg/application/... -run StreamVideo` PASS
- `grep -n "StreamVideoUseCase" src/go_api/pkg/{app,container,interfaces/connectrpc}` → wired in all three

## Context file
Maintain `.prompts/context/webrtc-migration-step-5-go-video.md`. Record:
- `pion/webrtc` version chosen and why
- How signaling reuses `WebRTCSignalingService` (SDP offer source, ICE candidate exchange)
- Label convention for the Go→C++ DataChannel (must be communicated to Agent D)
- Test coverage summary

Follow template §7 of the shared prompt.

## Done when
- New files created, tests pass, handler wired.
- `go build` and `go vet` clean.
- DataChannel label convention documented for Agent D.
- Step 5 row in INDEX marked `[x]`.
