# Step 3: Remove WebSocket Feature Flag and Stream Config

## Goal
Finish the stream-config cleanup by removing WebSocket transport remnants and returning a single WebRTC signaling endpoint from config.

## Progress Table
| Task | Status | Notes |
|---|---|---|
| Confirm `ws_transport_format` is removed | [x] | `config/flags.goff.yaml` has no such flag. |
| Confirm deleted stream-config files stay removed | [x] | `get_stream_config_use_case*` and `stream_config.go` do not exist. |
| Align `GetStreamConfig` with WebRTC-only contract | [x] | Returns type `webrtc` and reads endpoint from config. |
| Add config-backed signaling endpoint | [x] | Added `server.webrtc_signaling_endpoint` to config model, defaults, and YAMLs. |
| Run Go verification for cleanup | [x] | `go build ./...` and `go vet ./...` passed. |

## Files Touched
- `src/go_api/pkg/interfaces/connectrpc/config_handler.go` — read, edited
- `src/go_api/pkg/config/server_config.go` — read, edited
- `src/go_api/pkg/config/config.go` — read, edited
- `config/config.yaml` — read, edited
- `config/config.dev.yaml` — read, edited
- `config/config.staging.yaml` — read, edited
- `config/config.production.yaml` — read, edited

## Assumptions & Drift
- The repo no longer had a dedicated `StreamConfig` Go type or use case to read from.
- No existing config field exposed a WebRTC signaling endpoint, so a new `server.webrtc_signaling_endpoint` setting was added to satisfy the plan requirement that the endpoint come from config.
- The signaling procedure path was set to `/cuda_learning.WebRTCSignalingService/SignalingStream`, matching generated ConnectRPC procedure names.

## Verification Log
### Local checks
- `rg -n 'ws_transport_format' config/flags.goff.yaml` → PASS (no matches)
- `rg -n '\bStreamConfig\b|\bGetStreamConfigUseCase\b|WithGetStreamConfigUseCase|getStreamConfigUC' src/go_api` → PASS (no code matches; README still contains historical docs)
- `rg -n 'webrtc-signaling|/webrtc\.SignalingService/Signaling|webrtc_signaling_endpoint' {src/go_api/**/*.go,config/*.yaml}` → PASS (`webrtc-signaling` and old path removed; new config-backed endpoint present)
- `cd src/go_api && go build ./...` → PASS
- `cd src/go_api && go vet ./...` → PASS

### Global checks (if run)
- Not run in this step context.

## Open Questions / Escalations
- `src/go_api/README.md` still references `GetStreamConfigUseCase` and `StreamConfig` as historical documentation. Left unchanged.

## Attempt Log
### Attempt 1 — 2026-04-14T23:08:27.403Z
- Verified the feature flag and deleted stream-config files were already removed.
- Reworked `ConfigHandler.GetStreamConfig` to return a single `webrtc` endpoint backed by config.
- Added `server.webrtc_signaling_endpoint` defaults and config entries, then rebuilt and vetted the Go API successfully.
