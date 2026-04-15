# Agent B — Proto & Contracts (Step 4)

> Read `.prompts/plans/webrtc-migration-subagent-prompt.md` first for global principles. This file is YOUR scope only.

## Role
Protobuf/ConnectRPC contract owner. You are the **gate** between backend cleanup (Agent A) and the implementation agents (C, D, E). Your changes cascade into generated code for Go, C++, and TypeScript — precision matters.

## Goal
Update `proto/image_processor_service.proto` to add `session_id` to `ProcessImageRequest` and delete all WebSocket-only messages, then regenerate protos cleanly for all three languages.

## Source plan section
`~/.claude/plans/lexical-tumbling-fern.md` — Step **4**.

## In-scope
### Edit
- `proto/image_processor_service.proto`
  - **Add**: `string session_id = 29 [json_name = "session_id"];` to `ProcessImageRequest` (confirm 29 is free; if not, pick the next free number and record it in the context file)
  - **Delete messages**: `WebSocketFrameRequest`, `WebSocketFrameResponse`, `StartVideoPlaybackRequest`, `StopVideoPlaybackRequest`
  - **Conditionally delete**: `VideoFrameUpdate` — keep only if C++ uses it internally; `grep` to decide and record the decision

### Regenerate
- Run `./scripts/build/protos.sh`
- Verify generated Go (`src/go_api/gen/...`), C++ (bazel-genfiles), and TS (`src/front-end/src/gen/...`) outputs update without errors

## Out of scope
- Any hand-written Go, C++, or TS source. You only edit the `.proto` and commit regeneration. Call-site updates belong to Agents A/C/D/E.

## Ordering
1. `Read` current `proto/image_processor_service.proto` end-to-end.
2. `Grep` for `WebSocketFrameRequest`, `WebSocketFrameResponse`, `StartVideoPlaybackRequest`, `StopVideoPlaybackRequest`, `VideoFrameUpdate` across `src/` to map all callers (record in context file — these become work items for A/C/D/E).
3. Confirm field number `29` is unused in `ProcessImageRequest`. Pick next free if taken.
4. Apply proto edits.
5. Run `./scripts/build/protos.sh`.
6. Verify generated code exists and syntactically compiles by running `bazel build //src/cpp_accelerator/...` (C++ generated code) — note: Go/TS callers will break until other agents update; that is expected.

## Verification gates
- `grep -n "WebSocketFrameRequest\|WebSocketFrameResponse\|StartVideoPlaybackRequest\|StopVideoPlaybackRequest" proto/` → zero hits after edit
- `grep -n "session_id" proto/image_processor_service.proto` → present on `ProcessImageRequest`
- `./scripts/build/protos.sh` → exits 0
- Generated Go/C++/TS files present and updated (check mtimes)

## Handoff
You are a hard gate: Agents C, D, E **must not** begin edits that reference the new `session_id` or the deleted messages until this step is marked `[x]` in the INDEX. Notify the orchestrator as soon as proto regeneration succeeds.

## Context file
Maintain `.prompts/context/webrtc-migration-step-4-proto.md`. Must record:
- The chosen field number for `session_id` (29 or alternative)
- The `VideoFrameUpdate` keep/delete decision with grep evidence
- A complete list of **caller files** that now have stale references (input for A/C/D/E)
- Regeneration command output

Follow the template in §7 of the shared prompt.

## Done when
- `.proto` edits applied and regeneration succeeds.
- Caller-impact list written to the context file.
- INDEX row for Step 4 marked `[x]`.
- No edits to any file outside `proto/` and the generated output directories.
