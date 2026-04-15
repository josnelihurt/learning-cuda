# Lessons Learned — cpp_accelerator WebRTC Pipeline

## Bug #1: `mTracks` weak_ptr expires before `addTrack()` runs

### What happened
When processing the browser's SDP offer, libdatachannel's `processRemoteDescription` creates each track as a local `shared_ptr<impl::Track>` inside an `if` block. When the block exits, the local goes out of scope. `mTracks["1"]` only stores a `weak_ptr` → it expires immediately.

When `addTrack(media)` is later called for the outbound track (mid=1), internally it does `mTracks["1"].lock()` → fails → creates a new impl BUT `mTracks.emplace("1", new)` is a no-op (key already exists). The new impl never enters `mTracks`. `openTracks()` finds the expired weak_ptr → skips mid=1 → **the outbound track never opens**.

### Symptom
`"Outbound processed video track opened (mid=1)"` never appeared in logs.

### Fix
In the `onTrack` callback, capture the SendOnly track (mid=1) into `session->outbound_video_track` **before** any other processing. This keeps the `shared_ptr<impl::Track>` alive so that `mTracks["1"].lock()` succeeds when `addTrack()` looks it up.

```cpp
// In onTrack callback, BEFORE processing mid:
if (track->direction() == rtc::Description::Direction::SendOnly) {
    session->outbound_video_track = track;  // keeps impl alive
}
```

### Why it matters
`libdatachannel 0.24.0` uses `unordered_map<string, weak_ptr<impl::Track>>` — never a `shared_ptr` directly. The responsibility of keeping track impls alive falls **entirely** on user code via the `rtc::Track` objects returned from `onTrack` and `addTrack`.

---

## Bug #2: `incomingChain` execution order is the OPPOSITE of what you'd expect

### What happened
`MediaHandler::incomingChain` processes the chain in **reverse order**: it calls the `next` handler first, then the current handler:

```cpp
void MediaHandler::incomingChain(message_vector &messages, const message_callback &send) {
    if (auto handler = next())
        handler->incomingChain(messages, send);  // NEXT first
    incoming(messages, send);                    // CURRENT after
}
```

This is the opposite of `outgoingChain`, which processes current first, then next.

We had the chain configured as:
```cpp
// WRONG
session->inbound_rtcp_session->addToChain(session->inbound_depacketizer);
track->setMediaHandler(session->inbound_rtcp_session);
// Chain: rtcp_session → depacketizer
// Actual incoming() execution order: depacketizer first, rtcp_session second
```

**Result:** `H264RtpDepacketizer` received the raw RTP packet, assembled it into an H264 frame, and returned it. Then `RtcpReceivingSession::incoming` saw the assembled H264 frame (which starts with `0x00 0x00 0x00 0x01`) and tried to validate it as an RTP packet: the version byte was `0x00` (not `2`) → **silently dropped the frame**.

```
[Handler chain] raw RTP → depacketizer → H264 frame → rtcp_session rejects H264 (bad RTP version) → messages = []
```

### Symptoms
- The depacketizer successfully assembled frames (`result.size=1` in logs)
- But `onFrame` never fired
- `mRecvQueue` never received anything
- No exceptions, no error logs anywhere

### Fix
Reverse the chain setup so the execution order is correct:

```cpp
// CORRECT
// incomingChain executes: next first → current after
// We want: rtcp_session first (validate RTP), depacketizer second (assemble frame)
// → depacketizer is the root, rtcp_session is next
session->inbound_depacketizer->addToChain(session->inbound_rtcp_session);
track->setMediaHandler(session->inbound_depacketizer);
// Actual incoming() execution order: rtcp_session first ✅, depacketizer second ✅
```

### General rule for libdatachannel MediaHandler chains

| Direction      | Execution order in `*Chain()` |
|----------------|-------------------------------|
| `outgoingChain` | current → next → next... (left to right) |
| `incomingChain` | ...next → next → current (right to left, reversed) |

This design means that **a single shared chain** (e.g., `Packetizer → SrReporter`) works symmetrically for both encode (outgoing: packetize first) and decode (incoming: sr-report first, then depacketize). However, if you define **separate** chains for incoming and outgoing, you must configure them in the **reverse** order of desired execution.

### How to detect this class of bug
Add a log in `VideoRtpDepacketizer::incoming` showing `result.size` at the end. If the depacketizer produces frames but they never reach `onFrame`, look for a handler running **after** the depacketizer in the chain (in execution order) that silently discards the assembled output.

---

## Diagnostic methodology that worked

Work through the pipeline layer by layer, adding targeted logs to confirm or rule out each stage:

1. **Confirm packet routing**: log in `dispatchMedia` to verify SSRC → track mapping.
2. **Confirm `track->incoming()` is called**: log weak_ptr validity in `dispatchMedia`.
3. **Confirm handler chain runs**: log before `incomingChain()` call in `Track::incoming`.
4. **Confirm depacketizer output**: log `result.size` at end of `VideoRtpDepacketizer::incoming`.
5. **Confirm frames reach `mRecvQueue`**: log before `mRecvQueue.push()` in `Track::incoming`.

The gap between step 4 (frames produced) and step 5 (nothing in queue) pinpointed the failure to the code running **between** the depacketizer and the queue — which turned out to be `RtcpReceivingSession` running after the depacketizer due to the reversed chain order.
