# Step 2: Remove WebSocket Package (Go)

## Goal
Remove the remaining Go application wiring that only existed for the deleted WebSocket package.

## Progress Table
| Task | Status | Notes |
|---|---|---|
| Confirm `pkg/interfaces/websocket/` is deleted | [x] | Directory no longer exists. |
| Verify websocket setup methods are gone from `app.go` | [x] | `setupWebSocketHandler` and `setupWebRTCSignalingWebSocket` are absent. |
| Remove dead `grpcProcessor` app wiring | [x] | Removed `App.grpcProcessor`, `WithGRPCProcessor`, and its use in `cmd/server/main.go`. |
| Run Go verification for cleanup | [x] | `go build ./...` and `go vet ./...` passed. |

## Files Touched
- `src/go_api/pkg/app/app.go` — read, edited
- `src/go_api/cmd/server/main.go` — read, edited

## Assumptions & Drift
- The WebSocket package had already been deleted before this attempt.
- `evaluateFFUC` is still used by ConnectRPC config wiring, so it was intentionally kept.

## Verification Log
### Local checks
- `rg -n '\bWithGRPCProcessor\b|\bgrpcProcessor\b|setupWebSocketHandler|setupWebRTCSignalingWebSocket' src/go_api` → PASS (no matches after edit)
- `cd src/go_api && go build ./...` → PASS
- `cd src/go_api && go vet ./...` → PASS

### Global checks (if run)
- Not run in this step context.

## Open Questions / Escalations
- `src/go_api/README.md` still documents deleted WebSocket and stream-config concepts. This was left untouched because Step 2/3 scope is code/config cleanup, not docs.

## Attempt Log
### Attempt 1 — 2026-04-14T23:08:27.403Z
- Confirmed the WebSocket package and setup methods were already gone.
- Removed the remaining dead `grpcProcessor` field and constructor option from the Go app wiring.
- Rebuilt and vetted the Go API successfully.
