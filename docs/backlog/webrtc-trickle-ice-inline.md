# Trickle ICE Inline in StartSession

## Context

This document describes an improvement path ("Fix 1 Option B") discussed but not implemented
during the WebRTC reliability work that addressed signaling race conditions and session cleanup
timeouts.

The problem being solved is the same race condition addressed by Option A (client-side buffering
of local ICE candidates until `StartSession` returns a SDP answer). Option A was chosen because
it is contained to the frontend and does not touch the proto contract.

This document captures why Option B remains attractive as a follow-up and what shipping it would
require.

## Problem Recap

The current negotiation flow is half-trickle:

1. Frontend creates offer, sets local description (triggers ICE gathering).
2. Frontend posts `StartSession` with the offer SDP.
3. Go API forwards to C++ gRPC, C++ builds SDP answer, returns through the chain.
4. Frontend applies remote description.
5. Meanwhile every local ICE candidate is posted as a separate `SendIceCandidate` HTTP call.
6. Remote candidates are polled through `PollEvents`.

ICE candidates discovered during step 1 can fire before step 2 completes on the server side.
Option A mitigates by buffering locally and flushing after `StartSession` resolves.

Even with Option A, the first candidates still cross the wire in separate round-trips. On a
high-latency path (browser -> vultr -> jetson over `xb19042.glddns.com:60061`) this adds
serialization overhead and increases the chance of ICE check failures on the first connectivity
check window.

## Proposal

Bundle the first N local ICE candidates inline with the `StartSession` request and the SDP
answer response, and only fall back to `SendIceCandidate` for candidates discovered after the
initial round-trip.

### Proto changes

```proto
message StartSessionRequest {
  string session_id = 1;
  string sdp_offer = 2;
  repeated IceCandidate initial_candidates = 3;  // new
  TraceContext trace_context = 4;
}

message StartSessionResponse {
  string session_id = 1;
  string sdp_answer = 2;
  repeated IceCandidate initial_candidates = 3;  // new
  TraceContext trace_context = 4;
}
```

Generated artifacts to regenerate:

- `proto/gen/*.pb.go`
- `proto/gen/genconnect/*.connect.go`
- `src/front-end/src/gen/webrtc_signal_pb.ts`
- `src/front-end/src/gen/webrtc_signal_connect.ts`
- C++ proto headers (bazel target `//proto:image_processor_service_proto`)

Run: `./scripts/build/protos.sh`

### Frontend changes

In `src/front-end/src/infrastructure/connection/webrtc-service.ts::negotiateSession`:

1. Wait briefly for ICE gathering to progress (e.g. until `iceGatheringState === 'gathering'`
   plus a short grace window of 50-100 ms, or until at least one host candidate is collected).
2. Include the collected candidates as `initial_candidates` in the `StartSession` request.
3. After applying the SDP answer, register any `initial_candidates` returned by the server via
   `peerConnection.addIceCandidate`.
4. Keep the existing `onicecandidate` + `SendIceCandidate` path for candidates discovered after
   the initial round-trip (server reflexive from STUN typically arrives later).

Remove the local buffer added in Option A, or keep it for safety; either works.

### Go changes

In `src/go_api/pkg/interfaces/connectrpc/webrtc_session_manager.go::StartSession`:

1. Forward `initial_candidates` to the C++ signaling stream as `IceCandidate` messages
   immediately after the `StartSession` message and before waiting for
   `StartSessionResponse`.
2. Collect any `ice_candidate` messages emitted by C++ before `start_session_response` arrives
   and embed them in the `StartSessionResponse.initial_candidates` returned to the frontend.

### C++ changes

In `src/cpp_accelerator/ports/grpc/webrtc_manager.cpp::CreateSession`:

1. After `setRemoteDescription`, apply any `initial_candidates` from the request (use the same
   path as `HandleRemoteCandidate`, respecting the pending queue if remote description is not
   fully processed yet).
2. After `setLocalDescription(Answer)`, drain `local_candidates_queue` up to the moment the
   answer is published and include those candidates in the response payload.

## Benefits

- Eliminates the race window entirely, not just the client-side symptom.
- Saves at least one RTT per ICE candidate gathered during the initial window (typically 2-4
  host candidates per session).
- Improves time-to-connect measurably on high-latency deployments.

## Trade-offs

- Touches all three services (Go, C++, TS) and the proto contract. Requires coordinated
  rollout: servers must be updated before clients start including `initial_candidates`, or the
  field must be additive (which it is, given proto3 semantics) so old servers silently ignore
  it.
- The C++ side must be careful to not block waiting for ICE gathering before replying; the
  `StartSession` response already races against gathering state. Any implementation must pick a
  time budget (e.g. 50 ms) and return whatever is ready.
- Adds a small state machine for "initial candidates already delivered" to avoid duplicates on
  the `PollEvents` path.

## When to Do It

Priority is low as long as Option A (client-side buffering) is holding up in production. Revisit
if any of the following happens:

- Sustained ICE connectivity failures on first attempt (>2% of sessions).
- Time-to-first-frame regresses when deploying to higher-latency regions.
- A TURN server is introduced (#511) and the extra per-candidate RTT becomes noticeable with
  TURN relay candidates.

## Related

- `#511 Study STUN/TURN for NAT traversal`
- `docs/backlog/webrtc-session-lifecycle-hardening.md`
