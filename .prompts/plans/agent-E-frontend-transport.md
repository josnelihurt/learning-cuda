# Agent E — Frontend WebRTC Transport (Steps 7 & 8-partial)

> Read `.prompts/plans/webrtc-migration-subagent-prompt.md` first. This file is YOUR scope only.

## Role
Senior TypeScript / Lit engineer owning the frontend transport layer. You finish implementing `WebRTCFrameTransportService`, remove the WebSocket transport from the frontend DI, and update the `IFrameTransportService` interface to use the proto types directly (the WS wrapper types are gone after Agent B).

## Goal
Make `WebRTC` the **only** frame transport on the frontend: webcam/image frames go via DataChannel, video playback is triggered via ConnectRPC `StartVideoPlayback` with the browser's active WebRTC `session_id`. Remove all WS code, types, and DI wiring.

## Source plan sections
`~/.claude/plans/lexical-tumbling-fern.md` — Step **7** and the frontend parts of Step **8**.

## Dependencies (hard gates)
- **Agent B** must have regenerated TS protos (deletes `WebSocketFrame*` types and adds `session_id` on `ProcessImageRequest`).
- Independent of Agents C and D for compilation; runtime verification of video obviously requires them.

## In-scope files
### Edit
- `src/front-end/src/infrastructure/transport/webrtc-frame-transport.ts` — implement all stubbed methods:
  - `connect()` — ensure `WebRTCService` peer + DataChannel are open
  - `sendFrameWithProcessingConfig(image, filters, accelerator, grayscale)` — build `ProcessImageRequest`, `toBinary()`, `dataChannel.send(buffer)`
  - `onFrameResult(callback)` — `dataChannel.onmessage` → `ProcessImageResponse.fromBinary()` → invoke callback
  - `sendStartVideo(videoId, filters, accelerator)` — call new `StartVideoPlayback` ConnectRPC endpoint, passing the active `session_id` from `WebRTCService`
  - `sendStopVideo()` — call `StopVideoPlayback` ConnectRPC
  - `isConnected()` — `dataChannel?.readyState === 'open'`
- `src/front-end/src/domain/interfaces/IFrameTransportService.ts` — replace `WebSocketFrameResponse` with `ProcessImageResponse`
- `src/front-end/src/infrastructure/transport/frame-transport-service.ts`
  - Remove `wsTransport` constructor parameter and all usages
  - `selectTransport()` → return `webrtcTransport` (no more `grpcTransport` fallback to WS)
  - Remove `wsTransport` fallback from `getCurrentTransport()`
  - Remove the WS branch of `onFrameResult` dual-registration
- Frontend DI wiring (wherever `FrameTransportService` is instantiated with 3 transports) — drop the WS argument
- Any remaining TS file importing `WebSocketFrameRequest`/`WebSocketFrameResponse` — remove import + usage

### Tests
- Update Vitest specs for the touched files; follow CLAUDE.md patterns (AAA, `sut`, `vi.fn()`, `vi.mock()`, `makeXXX`, `Success_/Error_/Edge_` naming)

## Out of scope
- Backend deletions (Agent A)
- Proto edits (Agent B)
- Go video pipeline (Agent C)
- C++ WebRTC (Agent D)

## Ordering
1. `Read` `webrtc-frame-transport.ts`, `frame-transport-service.ts`, `IFrameTransportService.ts`, `WebRTCService` (`connection/webrtc-service.ts`).
2. Confirm the regenerated TS proto client exposes `ProcessImageRequest`, `ProcessImageResponse`, `StartVideoPlayback`, `StopVideoPlayback`.
3. Implement `webrtc-frame-transport.ts`.
4. Update interface + remove WS from `frame-transport-service.ts` + DI.
5. Delete/update tests; remove any WS-specific specs.
6. `cd src/front-end && npm run build && npm run test`.

## Verification gates
- `grep -rn "WebSocketFrame\|wsTransport\|ws_transport_format" src/front-end/src` → zero hits
- `npm run build` PASS
- `npm run test` PASS
- Manual check: `FrameTransportService` constructor now takes only WebRTC + gRPC (or just WebRTC — confirm with plan intent)

## Context file
Maintain `.prompts/context/webrtc-migration-step-7-frontend.md`. Record:
- Final `FrameTransportService` constructor signature
- How `session_id` is obtained from `WebRTCService` for the `StartVideoPlayback` call
- Test coverage summary and any deleted test files

Follow template §7 of the shared prompt.

## Done when
- `npm run build` + `npm run test` pass.
- All `WebSocketFrame*` / `wsTransport` references gone from `src/front-end/src`.
- Step 7 row in INDEX marked `[x]`; the frontend items in the Step 8 cleanup row noted as done.
