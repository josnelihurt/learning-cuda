# Step 1: Delete `cgo_api.cpp` and BUILD target

## Goal
Confirm the legacy C++ CGO port is gone and clear the remaining in-scope backend references so Agent A's cleanup gate is satisfied.

## Progress Table
| Task | Status | Notes |
|---|---|---|
| Confirm `src/cpp_accelerator/ports/cgo/` is deleted | [x] | Directory is absent. |
| Verify no Bazel targets depend on `cgo_api` | [x] | No matches remain under `src/cpp_accelerator/`. |
| Remove stale in-scope `cgo` / WebSocket cleanup references | [x] | Cleaned comments and READMEs that still mentioned removed backend components. |
| Run cleanup verification | [x] | `go mod tidy`, `go build`, `go vet`, and `bazel build //src/cpp_accelerator/...` passed. |

## Files Touched
- `src/cpp_accelerator/application/commands/command_factory.cpp` — read, edited
- `src/cpp_accelerator/README.md` — read, edited
- `src/go_api/pkg/interfaces/connectrpc/handler.go` — read, edited
- `src/go_api/README.md` — read, edited

## Assumptions & Drift
- The CGO port had already been deleted before this attempt, so Step 1 was completion/bookkeeping plus cleanup of stale backend references.
- `go mod tidy` kept `github.com/gorilla/websocket` in the root module because acceptance tests outside Agent A's scope still import it.

## Verification Log
### Local checks
- `rg -n 'websocket|ws_transport_format|cgo_api|\\bStreamConfig\\b|\\bGetStreamConfigUseCase\\b' src/go_api src/cpp_accelerator config` → PASS (no in-scope matches after cleanup)
- `cd /home/jrb/code/cuda-learning && go mod tidy` → PASS
- `cd /home/jrb/code/cuda-learning/src/go_api && go build ./...` → PASS
- `cd /home/jrb/code/cuda-learning/src/go_api && go vet ./...` → PASS

### Global checks (if run)
- `cd /home/jrb/code/cuda-learning && bazel build //src/cpp_accelerator/...` → PASS

## Open Questions / Escalations
- None for Step 1.

## Attempt Log
### Attempt 1 — 2026-04-14T23:16:31.143Z
- Confirmed the CGO port was already absent.
- Removed the remaining in-scope backend references to deleted WebSocket/CGO pieces from code comments and READMEs so Agent A's grep gate matched the plan.
- Re-ran Go and C++ verification successfully.
