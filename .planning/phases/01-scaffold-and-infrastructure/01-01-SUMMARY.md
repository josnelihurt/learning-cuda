---
phase: 01-scaffold-and-infrastructure
plan: 01
subsystem: infra
tags: vite, react, typescript, mpa, webrtc

requires:
  - phase: None
    provides: Existing front-end in front-end/
provides:
  - Vite MPA with index.html and react.html plus manifest keys
  - Dev and preview middleware for GET /react and GET /lit
  - WebRTC Vitest import smoke using test-setup stubs
  - Bounded HTTPS dev smoke script (45s) for 127.0.0.1:3000
affects: 02-react-hooks, 03-react-components

tech-stack:
  added: []
  patterns:
    - prettyFrontendRoutesPlugin rewrites exact pathnames only
    - Preview server HTTP on 4173 for Playwright while dev stays HTTPS on 3000

key-files:
  created:
    - front-end/src/react/webrtc-stub-import.test.tsx
    - front-end/scripts/verify-vite-dev-routes.sh
  modified:
    - front-end/vite.config.ts
    - front-end/src/infrastructure/connection/webrtc-service.ts

key-decisions:
  - "Exported WebRTCService class for focused Vitest without mocking webrtc-service module"
  - "Dev smoke checks react.html markers (#root, entry script) not hydrated React text"

patterns-established:
  - "Exact-path /react and /lit rewrites to fixed HTML entries (ASVS-friendly)"

requirements-completed: [SCAF-02]

duration: 25min
completed: 2026-04-12
---

# Phase 1 Plan 01: Vite MPA and dev routes

**Outcome:** Vite builds both Lit and React HTML entries with manifest keys; dev/preview serve pretty `/react` and `/lit`; WebRTC-using modules construct under Vitest; a 45s-capped script proves dual dev routes when TLS certs exist.

## Performance

- **Tasks:** 4 (committed across 3 commits)
- **Files modified:** 4

## Task Commits

1. **Task 1: WebRTC Vitest smoke** — `79d7f6b` (test)
2. **Tasks 2–3: MPA header, pretty routes, preview HTTP** — `985ec43` (feat)
3. **Task 4: Bounded dev verify script** — `5940d79` (chore)

## Files Created/Modified

- `front-end/vite.config.ts` — MPA comment, `prettyFrontendRoutesPlugin`, `preview.https: false` for e2e
- `front-end/src/react/webrtc-stub-import.test.tsx` — constructs `WebRTCService` with stubs
- `front-end/src/infrastructure/connection/webrtc-service.ts` — `export class WebRTCService`
- `front-end/scripts/verify-vite-dev-routes.sh` — `timeout 45s` wrapper, curl probes

## Verification

- `npx vitest run src/react/webrtc-stub-import.test.tsx`
- `npm run build` with `dist/index.html`, `dist/react.html`, manifest keys
- `front-end/scripts/verify-vite-dev-routes.sh` (requires `.secrets/` dev certs)

## Self-Check: PASSED
