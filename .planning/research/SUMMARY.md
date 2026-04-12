# Research Summary — React Frontend Migration

**Synthesized from:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md
**Date:** 2026-04-12

---

## Executive Summary

This project adds a React frontend as a parallel route (`/react/*`) to an existing Go + C++/CUDA platform that already has a working Lit Web Components frontend. The migration is not a rewrite — it is a feature-parity exercise running both SPAs from the same Vite build and same Go HTTP server, sharing all backend infrastructure (ConnectRPC, WebSocket, gRPC). The recommended approach is Vite MPA mode with dual entry points, React 19 + React Router v7, React Context for global state, and custom hooks that wrap existing TypeScript infrastructure services without duplicating them.

The greatest complexity is not React itself — it is the WebRTC streaming path. The `useWebRTCStream` hook must manage a full RTCPeerConnection lifecycle (idle → connecting → connected → failed → reconnecting), correctly bypass React state for frame rendering (drawing directly to canvas via `useRef`), and clean up every resource on unmount. A project learning goal is React Context patterns, so Zustand should not be introduced until Context is demonstrably insufficient.

The top risks are all preventable if the scaffold phase is done correctly: Vite MPA config must be validated before any component work begins, the test environment needs WebRTC API stubs before the first React test, and the ConnectRPC singleton must be established as a Context-injected dependency (not imported from the DI container) before any hook is written.

---

## Key Findings

### Stack Additions
- **React 19.2.5** — current stable, no breaking changes for this use case
- **react-router 7.11.0** — `react-router-dom` is deprecated in v7; use `react-router` package
- **`@vitejs/plugin-react@^4`** — Vite 5 compatible (v6 targets Vite 8, do not use)
- **`@testing-library/react@^16.3.2`** — React 19 support; requires `@testing-library/dom` as explicit peer dep
- **Do NOT add:** Next.js, Redux, Zustand (yet), react-query, CSS-in-JS

### Features (P1 — parity required)
- ConnectRPC client singleton + `useAsyncGRPC` hook
- Filter selection panel with live preview
- Image upload with XMLHttpRequest (not fetch) for progress tracking
- Canvas frame display via `useRef` — zero React state in render loop
- `useWebRTCStream` with full RTCPeerConnection lifecycle management
- Video source selector
- Toast notification system
- Health monitor display
- gRPC error modal
- File management list

### Architecture
- **Vite MPA** with dual `rollupOptions.input`: `lit` and `react` entry points
- Output to `static/js/dist/lit/` and `static/js/dist/react/`
- Go server routes `/react/*` → `index-react.html`, `/lit/*` and `/` → `index-lit.html`
- `src/gen/` and `src/infrastructure/` are shared (plain TypeScript, no Lit coupling)
- Lit DI container is NOT shared — React replaces it with Context providers in `react-main.tsx`

---

## Top Pitfalls

1. **`srcObject` as JSX prop silently fails** — always use `useRef` + `useEffect`
2. **Vite MPA 404 in dev** — structure HTML files correctly, validate routing before any component work
3. **React Context render storms during video** — `MediaStream`/frames go in `useRef`, never state or Context
4. **WebRTC leaks from missing `useEffect` cleanup** — every WebRTC effect must return cleanup; verify with camera LED + `chrome://webrtc-internals`
5. **Vitest happy-dom lacks WebRTC APIs** — stub `RTCPeerConnection` and `navigator.mediaDevices` in `test-setup.ts` before first React test
6. **`@vitejs/plugin-react` not added to `vitest.config.ts`** — add to both configs; run full Lit test suite after to confirm no regressions

---

## Roadmap Implications

**Suggested phases: 5**

| Phase | Name | Focus |
|-------|------|-------|
| 1 | Scaffold and Infrastructure | Vite MPA config, Go routing, React entry point, WebRTC test stubs, ServiceContext |
| 2 | Core Hook Infrastructure | `useAsyncGRPC`, `ToastContext`, `useHealthMonitor`, `useFilters`, `useFileList` |
| 3 | Static Feature UI | Filter panel, image upload, image display, file list, settings UI |
| 4 | Video Streaming and WebRTC | `useWebRTCStream`, canvas frame display, video source selector, error boundaries |
| 5 | Polish and Parity Validation | CSS audit, RTL coverage, dual-route parity check, production build, strict TS |

**Phase 1 is the highest-risk phase** — all 6 pitfalls above are preventable here. Do not compress it.

---

## Research Flags

**Needs deeper research (plan-phase):** Phase 4 (WebRTC hook state machine, ICE candidate queuing); Phase 1 (exact Vite MPA dev server config with Go proxy).

**Open questions to resolve in Phase 1:**
- Does `asset_manifest.go` use Vite manifest `name` field? Verify before Vite MPA config change
- Does `@vitejs/plugin-react` conflict with Lit's TypeScript decorator transform in shared `tsconfig.json`?
- Does Go `dev_server_paths` config need changes for Vite MPA?

**Open questions to resolve before Phase 4:**
- Exact WebSocket signaling protocol in Go server — read `pkg/interfaces/websocket/` before designing `useWebRTCStream`
- CSS portability: scan for Lit `:host` selectors before Phase 3 CSS reuse

---

## Confidence

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Versions verified via npm + official changelogs |
| Features | HIGH | Based on direct Lit codebase analysis |
| Architecture | HIGH | Based on direct analysis of `vite.config.ts`, `statichttp/`, `src/infrastructure/` |
| Pitfalls | HIGH | React bugs confirmed via open GitHub issues |

---
*Research complete: 2026-04-12*
