# Agent A — Backend Cleanup (Steps 1, 2, 3, 8-partial)

> Extended sub-agent prompt. Read the shared prompt `.prompts/plans/webrtc-migration-subagent-prompt.md` first for global operating principles, context-file protocol, and Definition of Done. This file is YOUR scope only.

## Role
Senior Go + C++/Bazel engineer. You delete legacy WebSocket transport and the CGO shim. You do **not** touch proto files, the Go video pipeline, C++ WebRTC processing, or the frontend.

## Goal
Remove all WebSocket and legacy CGO code from the Go API and C++ accelerator so the repository no longer contains any reference to `websocket`, `ws_transport_format`, `cgo_api`, `StreamConfig`, or `GetStreamConfigUseCase` — **except** for proto-generated files (Agent B owns those) and frontend code (Agent E owns that).

## Source plan sections
`~/.claude/plans/lexical-tumbling-fern.md` — Steps **1**, **2**, **3**, and the Go-side portion of Step **8**.

## In-scope files
### Delete
- `src/cpp_accelerator/ports/cgo/cgo_api.cpp` (and its BUILD target; remove `cgo/` dir if empty)
- `src/go_api/pkg/interfaces/websocket/` (entire package)
- `src/go_api/pkg/application/get_stream_config_use_case.go`
- `src/go_api/pkg/application/get_stream_config_use_case_test.go`
- `src/go_api/pkg/config/stream_config.go`
- `config/flags.goff.yaml` → remove `ws_transport_format` entry

### Edit
- `src/go_api/pkg/app/app.go` — drop `setupWebSocketHandler`, `setupWebRTCSignalingWebSocket`, `websocket` import, `grpcProcessor`, `evaluateFFUC` (if unused elsewhere), `getStreamConfigUC` field + `WithGetStreamConfigUseCase()` option
- `src/go_api/pkg/interfaces/connectrpc/config_handler.go` — replace WS `StreamEndpoint` in `GetStreamConfig` with a single WebRTC signaling endpoint; remove `getStreamConfigUseCase`
- `src/go_api/pkg/container/container.go` — remove `GetStreamConfigUseCase` wiring
- `src/go_api/pkg/interfaces/connectrpc/vanguard.go` — remove `GetStreamConfigUC` from `VanguardConfig` if only used for stream config
- `src/go_api/pkg/config/config.go` — remove `Stream StreamConfig` field
- `src/go_api/config/config.yaml` + dev/staging/production variants — remove `stream:` section
- `go.mod` / `go.sum` — `go mod tidy` after removing `gorilla/websocket` usages

## Out of scope (do NOT touch)
- `proto/image_processor_service.proto` → Agent B
- Any file under `src/front-end/` → Agent E
- `src/cpp_accelerator/ports/grpc/webrtc_manager.*` and `image_processor_service_impl.cpp` → Agent D
- New files for Go video pipeline (`stream_video_use_case.go`, `infrastructure/webrtc/go_peer.go`) → Agent C

## Ordering inside your scope
1. Delete `cgo_api.cpp` + BUILD target. Verify no Bazel target depends on it (`grep -r cgo_api src/cpp_accelerator/`).
2. Delete `src/go_api/pkg/interfaces/websocket/` entirely.
3. Edit `app.go` to remove all WS wiring. Compile must fail loudly until remaining edits land — that's fine.
4. Delete stream-config use case + config struct files.
5. Edit `config_handler.go`, `container.go`, `vanguard.go`, `config.go`, yaml configs.
6. Remove `ws_transport_format` from `flags.goff.yaml`.
7. `cd src/go_api && go mod tidy && go build ./... && go vet ./...`.
8. `bazel build //src/cpp_accelerator/...` to confirm cgo removal is clean.

## Verification gates
Local (run after each sub-step):
- `grep -rn "websocket\|ws_transport_format\|cgo_api\|StreamConfig\|GetStreamConfigUseCase" src/go_api src/cpp_accelerator config` → expect zero hits when done
- `cd src/go_api && go build ./...` → PASS
- `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server` → PASS

Handoff gate to Agent C: Agent C cannot start `StreamVideoUseCase` wiring until `app.go` and `container.go` compile cleanly in your branch.

## Context file
Maintain `.prompts/context/webrtc-migration-step-1-cgo-cleanup.md`, `.prompts/context/webrtc-migration-step-2-websocket-pkg.md`, `.prompts/context/webrtc-migration-step-3-stream-config.md`, and update `.prompts/context/webrtc-migration-INDEX.md` after each.

Each context file must follow the template in `.prompts/plans/webrtc-migration-subagent-prompt.md` §7 (Progress Table, Files Touched, Assumptions & Drift, Verification Log, Open Questions, Attempt Log).

## Done when
- All grep checks return zero hits across your scope.
- `go build`, `go vet`, and the `bazel build` target all pass.
- Three step context files marked `[x]` and INDEX updated.
- No edits outside the in-scope file list.
