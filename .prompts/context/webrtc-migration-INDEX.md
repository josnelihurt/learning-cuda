# WebRTC Migration Index

| Step | Title | Status | Context File | Notes |
|---|---|---|---|---|
| 1 | Delete `cgo_api.cpp` and BUILD target | done | `.prompts/context/webrtc-migration-step-1-cgo.md` | CGO port already absent; cleared remaining in-scope backend references and passed Go/C++ verification. |
| 2 | Remove WebSocket package (Go) | done | `.prompts/context/webrtc-migration-step-2-go-websocket-cleanup.md` | Removed lingering dead `grpcProcessor` app wiring; Go build/vet passed. |
| 3 | Remove WebSocket feature flag and stream config | done | `.prompts/context/webrtc-migration-step-3-config-cleanup.md` | `GetStreamConfig` now returns a config-backed `webrtc` signaling endpoint; Go build/vet passed. |
| 4 | Proto & contracts | done | `.prompts/context/webrtc-migration-step-4-proto.md` | `session_id = 29` added; WebSocket-only messages removed; proto regen and C++ build passed. |
| 5 | Go video pipeline via WebRTC | done | `.prompts/context/webrtc-migration-step-5-go-video.md` | Added Start/Stop playback RPCs to resolve contract drift, implemented `StreamVideoUseCase` + Go WebRTC peer, and passed Go build/vet/tests. |
| 6 | C++ WebRTC processing | done | `.prompts/context/webrtc-migration-step-6-cpp-webrtc.md` | `WebRTCManager` now processes binary proto frames, routes browser-session responses with `SendToSession`, and passed Bazel build/tests. |
| 7 | Frontend transport via WebRTC | done | `.prompts/context/webrtc-migration-step-7-frontend-transport.md` | Active React/Lit transport now uses the WebRTC data-channel path and frontend validation passed. |
| 8 | Remaining cleanup | blocked | `.prompts/context/webrtc-migration-step-8-cleanup.md` | Build/test verification passes, but browser signaling still uses WebSocket and acceptance fixtures still reference removed WebSocket frame contracts. |
