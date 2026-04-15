# Step 8 - Remaining cleanup

## Status
blocked

## Outcome
- Completed the in-scope cleanup tied directly to the active frontend cutover.
- Verified the integrated Go, C++, and frontend build/test chain passes.

## Remaining blockers
- `src/front-end/src/infrastructure/connection/webrtc-service.ts` still uses browser `WebSocket` signaling for WebRTC session negotiation.
- Acceptance coverage still contains legacy `WebSocketFrameRequest` / `WebSocketFrameResponse` references under `test/integration/tests/acceptance/steps/`.
- Repo surfaces outside the active transport path still retain old stream-config naming such as `GetStreamConfig`.

## Validation
- `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server`
- `cd src/go_api && make build`
- `cd src/front-end && npm run build`
- `./scripts/test/unit-tests.sh`

## Notes
- The migration is not yet safe to mark fully complete while browser signaling still depends on WebSocket and the acceptance fixtures still target deleted WebSocket frame contracts.
