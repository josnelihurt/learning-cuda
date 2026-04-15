# Agent D — C++ WebRTC Frame Processing (Step 6)

> Read `.prompts/plans/webrtc-migration-subagent-prompt.md` first. This file is YOUR scope only.

## Role
Senior C++/Bazel engineer working in the `cpp_accelerator` module. You turn the existing DataChannel "ping/pong" stub into a real frame-processing pipeline backed by `ProcessorEngine`, and add session-based routing so processed frames reach the correct browser DataChannel.

## Goal
Make `WebRTCManager` process incoming serialized `ProcessImageRequest` frames on every DataChannel and send serialized `ProcessImageResponse` back, AND add a `SendToSession()` helper so `StreamProcessVideo` (invoked by Agent C's Go video pipeline) can route processed frames to the correct browser session's DataChannel based on `request.session_id()`.

## Source plan section
`~/.claude/plans/lexical-tumbling-fern.md` — Step **6**.

## Dependencies (hard gates)
- **Agent B** must have completed proto regeneration — you need `session_id` on `ProcessImageRequest`.
- Independent of Agents A, C, E (you can start as soon as Agent B is `[x]`).

## In-scope files
### Edit
- `src/cpp_accelerator/ports/grpc/webrtc_manager.h`
  - Add `ProcessorEngine*` constructor parameter and member
  - Declare `void SendToSession(const std::string& session_id, const std::string& bytes)`
  - Declare internal session→DataChannel map (mutex-protected)
- `src/cpp_accelerator/ports/grpc/webrtc_manager.cpp`
  - Store `ProcessorEngine*`
  - On `DataChannel.onOpen`: register channel in session map keyed by peer session id
  - On `DataChannel.onClose`: remove from map
  - On `DataChannel.onMessage(binary)`:
    1. `ProcessImageRequest req; req.ParseFromString(msg)` — handle parse failure with logger
    2. `ProcessImageResponse resp; engine_->ProcessImage(req, &resp)`
    3. `std::string out; resp.SerializeToString(&out)`
    4. `channel->send(rtc::binary(out))`
  - Implement `SendToSession()` — lookup + send, log warning on miss
- `src/cpp_accelerator/ports/grpc/image_processor_service_impl.cpp`
  - In `StreamProcessVideo()`: after `engine_->ProcessImage(request, &response)`, if `request.session_id()` is non-empty → serialize response and call `webrtc_manager_->SendToSession(request.session_id(), bytes)`. Keep `stream->Write()` only if current callers still need it; otherwise remove.
  - Add `WebRTCManager*` dependency if not already present
- `src/cpp_accelerator/ports/grpc/server_main.cpp` (or wherever `WebRTCManager` is constructed)
  - Pass the already-initialized `ProcessorEngine*` into `WebRTCManager`
  - Pass `WebRTCManager*` into `ImageProcessorServiceImpl` if not already
- Corresponding `BUILD` files — add deps if needed

## Out of scope
- Proto edits (Agent B)
- Go / frontend / websocket cleanup (A, C, E)
- CUDA kernels, filters, `ProcessorEngine` internals — reuse as-is

## Ordering
1. `Read` `webrtc_manager.h/cpp`, `image_processor_service_impl.cpp`, `server_main.cpp` fully.
2. Confirm with Agent C's context file the DataChannel **label convention** used by the Go video peer — the session map must distinguish "browser" channels from "go-video" channels so Go frames don't get routed back to Go.
3. Update `webrtc_manager.h/cpp`: inject `ProcessorEngine`, add session map + `SendToSession`.
4. Replace the ping/pong stub with real frame processing.
5. Update `image_processor_service_impl.cpp` for session-based routing in `StreamProcessVideo`.
6. Update `server_main.cpp` wiring.
7. `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server`
8. `bazel test //src/cpp_accelerator/...` — fix any equivalence/unit tests broken by the new dependency

## Verification gates
- `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server` PASS
- `bazel test //src/cpp_accelerator/...` PASS
- `grep -n "ping\|pong" src/cpp_accelerator/ports/grpc/webrtc_manager.cpp` → zero hits in the onMessage path
- `grep -n "SendToSession" src/cpp_accelerator/ports/grpc/` → defined in `webrtc_manager` and called from `image_processor_service_impl`

## Thread safety
Session map must be mutex-protected. DataChannel callbacks run on libdatachannel threads; `ProcessorEngine::ProcessImage` must be safe to call from those threads (verify by reading `ProcessorEngine` — do not modify it; if it is not thread-safe, record as an escalation in the context file instead of papering over it).

## Context file
Maintain `.prompts/context/webrtc-migration-step-6-cpp-webrtc.md`. Record:
- DataChannel label convention adopted (must match Agent C)
- Session map ownership + locking strategy
- Any `ProcessorEngine` thread-safety concerns discovered
- Bazel test results

Follow template §7 of the shared prompt.

## Done when
- Bazel build + tests pass.
- `onMessage` processes real proto frames.
- `SendToSession` wired into `StreamProcessVideo`.
- Step 6 row in INDEX marked `[x]`.
