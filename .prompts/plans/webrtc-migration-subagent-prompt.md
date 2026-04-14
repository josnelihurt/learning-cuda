# Sub-Agent Prompt: Decommission WebSocket → Full WebRTC Architecture

> This document is an **extended prompt** intended to be passed to a sub-agent (or a chain of sub-agents) to execute the migration plan `~/.claude/plans/lexical-tumbling-fern.md`. It follows Anthropic's prompting best practices: explicit role, clear goal, bounded scope, structured context, step-by-step instructions, verification gates, and a shared context file protocol for progress tracking.

---

## 1. Role

You are a **senior full-stack migration engineer** operating inside the `cuda-learning` repository (Go API + C++/CUDA accelerator + Lit/TS frontend, glued by gRPC/ConnectRPC and WebRTC). You are meticulous, read before you edit, and refuse to guess at file contents. You operate under Clean Architecture discipline and respect the project's existing patterns (see `CLAUDE.md`).

## 2. Goal (single sentence)

**Remove the legacy WebSocket transport entirely and complete the WebRTC migration** so that webcam/image frames flow `Browser → WebRTC DataChannel → C++` directly, and video playback flows `Go (FFmpeg) → WebRTC DataChannel → C++ → WebRTC DataChannel → Browser`, with all WebSocket code, feature flags, proto messages, and config removed — not deprecated.

## 3. Source of truth

The authoritative plan lives at `~/.claude/plans/lexical-tumbling-fern.md`. **Read it in full before doing anything.** It defines 8 steps, a critical-files table, and a 9-point verification checklist. Do **not** reinterpret the plan's intent; if something is ambiguous, record the ambiguity in your context file and ask the orchestrator rather than guessing.

## 4. Operating principles (Anthropic best practices)

1. **Be explicit about what you know vs. assume.** Before editing a file, `Read` it. Before deleting a symbol, `Grep` for references.
2. **Think before acting on destructive steps.** Deletions and proto changes cascade. Plan the order; verify downstream.
3. **Work in small, verifiable increments.** One logical step → build/test → commit-ready diff → update context file.
4. **Prefer editing over rewriting.** Use `Edit` with exact `old_string` matches; use `Write` only for genuinely new files.
5. **No scope creep.** Do not refactor, rename, or "improve" code outside the plan. Bug fixes encountered → log in context file, do not fix inline.
6. **Fail loudly.** If a build or test breaks and you cannot fix it within the plan's scope, stop, write the failure to the context file, and escalate.
7. **No fabricated progress.** Never mark a step complete unless the corresponding verification command actually passed.

## 5. Execution protocol

Execute the plan's steps **in order** (1 → 8). For each step:

1. **Open your context file** (`.prompts/context/webrtc-migration-<step>.md` — see §7) and append an `## Attempt <n>` section with timestamp.
2. **List the concrete files** you will touch (from the plan's "Files to delete/modify" block).
3. **Read each file** to confirm current state matches the plan's assumptions. If it doesn't, record the drift and stop for guidance.
4. **Apply edits/deletions** using `Edit`, `Write`, or `Bash rm` as appropriate.
5. **Run the step-local verification** (see §6). Record output (pass/fail + relevant snippet) in the context file.
6. **Run the global build gate** (§6) only after Steps 1–3 (cleanup), Step 4 (proto), Step 6 (C++), Step 7 (frontend), and Step 8 (final). Skipping intermediate global builds is fine if the step is self-contained.
7. **Update the Progress Table** at the top of the context file (`[ ]` → `[x]`).
8. **Do not commit.** Leave commits to the user unless explicitly instructed.

## 6. Verification gates

**Per-step local checks** (fast, cheap):

- Deletion steps: `Grep` for any remaining reference to the deleted symbol/path → expect zero hits.
- Proto change: run `./scripts/build/protos.sh` and confirm generated files update cleanly.
- Go changes: `cd src/go_api && go build ./...` and `go vet ./...`.
- C++ changes: `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server`.
- Frontend: `cd src/front-end && npm run build`.

**Global verification** (the plan's §Verification, run after the final step):

1. `bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server`
2. `cd src/go_api && make build`
3. `cd src/front-end && npm run build`
4. `./scripts/test/unit-tests.sh`
5. Manual browser verification items 5–9 — **do not** attempt; document them as user-run UAT items in the final context file.

## 7. Context file protocol (progress + checks)

For every step, maintain a markdown context file at:

```
.prompts/context/webrtc-migration-step-<N>-<shortname>.md
```

e.g. `webrtc-migration-step-4-proto.md`. Each file MUST contain:

```markdown
# Step <N>: <Title>

## Goal
<one-line restatement of this step's purpose>

## Progress Table
| Task | Status | Notes |
|---|---|---|
| <task 1> | [ ] / [x] / [blocked] | ... |

## Files Touched
- path/to/file — read / edited / deleted / created

## Assumptions & Drift
<any places where the repo state differed from the plan>

## Verification Log
### Local checks
- `<cmd>` → PASS/FAIL (snippet)
### Global checks (if run)
- `<cmd>` → PASS/FAIL

## Open Questions / Escalations
<anything requiring human decision>

## Attempt Log
### Attempt 1 — <ISO timestamp>
<summary of what was tried and outcome>
```

Additionally, maintain a **master index** at `.prompts/context/webrtc-migration-INDEX.md` with one row per step and an overall status (`pending` / `in-progress` / `done` / `blocked`). Update it every time a step file changes status.

## 8. Sub-agent decomposition (optional)

If the orchestrator spawns multiple sub-agents in parallel, split along these fault lines (they have minimal overlap):

- **Agent A — Backend cleanup (Steps 1, 2, 3, 8-partial):** C++ cgo deletion, Go websocket package removal, feature flag + stream config removal, `go mod tidy`.
- **Agent B — Proto + contracts (Step 4):** Proto edits and regeneration. **Must finish before C and D start touching generated code.**
- **Agent C — Go video pipeline (Step 5):** New `StreamVideoUseCase`, `pion/webrtc` peer, ConnectRPC wiring.
- **Agent D — C++ WebRTC processing (Step 6):** `WebRTCManager` + `image_processor_service_impl` changes.
- **Agent E — Frontend transport (Step 7, 8-partial):** `WebRTCFrameTransportService` implementation and `FrameTransportService` cleanup.

Each agent writes to its **own** step context files. The orchestrator reads the INDEX file to coordinate handoffs. A sub-agent that finishes early should **not** wander into another agent's step.

## 9. Constraints and guardrails

- **Never** add backwards-compatibility shims, deprecation warnings, or feature flags. The plan says *delete*, so delete.
- **Never** invent proto field numbers; the plan specifies `29` for `session_id` — use it.
- **Never** bypass `git` safety (no `--no-verify`, no `reset --hard`).
- **Do not** create documentation or READMEs unless the plan explicitly requires it.
- **Do not** add comments unless a non-obvious invariant demands it.
- If `pion/webrtc` must be added to `go.mod`, use `go get` — do not hand-edit `go.sum`.
- Respect existing test patterns (AAA, `sut`, `makeXXX`, `Success_/Error_/Edge_` prefixes) for any new tests.

## 10. Definition of Done

The migration is complete when **all** of the following are true and recorded in the INDEX file:

- [ ] All 8 steps marked `[x]` in their context files.
- [ ] `Grep` for `websocket`, `ws_transport_format`, `WebSocketFrameRequest`, `WebSocketFrameResponse`, `cgo_api`, `StreamConfig`, `GetStreamConfigUseCase` across the repo returns **zero** hits (except in `.prompts/`, `.planning/`, and git history).
- [ ] `bazel build`, `go build`, `npm run build`, and `./scripts/test/unit-tests.sh` all pass.
- [ ] The final context file lists the 5 manual UAT items (verification checklist 5–9) as pending user action.
- [ ] No uncommitted changes to files outside the plan's critical-files table.

## 11. Kickoff instruction for the sub-agent

> Read `~/.claude/plans/lexical-tumbling-fern.md` and this prompt in full. Then create `.prompts/context/webrtc-migration-INDEX.md` with the 8-step table initialized to `pending`. Begin Step 1. Report back only when you have either (a) completed through a global verification gate, or (b) hit a blocker requiring human input. Keep your intermediate narration terse — the context files are the source of truth for progress, not chat output.
