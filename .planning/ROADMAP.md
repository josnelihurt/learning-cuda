# Roadmap: CUDA Learning Platform — React Frontend Migration

## Overview

Five phases take the project from zero React code to a production-ready frontend with full feature parity to the existing Lit frontend. Phase 1 establishes the Vite MPA scaffold and static/proxy routing (Nginx + Traefik in production, Vite + Go split ports in dev) so both frontends coexist from day one. Phase 2 builds the shared hook and context infrastructure every component depends on. Phase 3 implements all static UI features (image processing, file management, settings, health). Phase 4 adds the WebRTC video streaming path — the most complex part. Phase 5 validates parity between the two routes and closes any gaps.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Scaffold and Infrastructure** - Vite MPA config, Go dual routing, React entry point, and WebRTC test stubs
- [ ] **Phase 2: Core Hook Infrastructure** - ConnectRPC context, async gRPC hook, toast, filters, and health monitor hooks
- [ ] **Phase 3: Static Feature UI** - Image processing, file management, settings, and health monitoring UI
- [ ] **Phase 4: Video Streaming and WebRTC** - Full WebRTC stream lifecycle, canvas frame display, video source selector
- [ ] **Phase 5: Polish and Parity Validation** - CSS audit, parity verification between /react and /lit routes

## Phase Details

### Phase 1: Scaffold and Infrastructure
**Goal**: Developer can load the React frontend at `/react` and the Lit frontend at `/lit` simultaneously from a single Vite MPA build; production serves both routes on one user-facing origin via Nginx (with Go for API/WebSocket behind the same host), and local dev uses Vite for UI plus Go for backend per REQUIREMENTS SCAF-01
**Depends on**: Nothing (first phase)
**Requirements**: SCAF-01, SCAF-02
**Success Criteria** (what must be TRUE):
  1. Developer visits `/react` in the browser and gets a React app shell (not a 404 or the Lit page)
  2. Developer visits `/lit` and gets the existing Lit frontend unchanged
  3. `npm run dev` and `npm run build` both succeed with dual entry points producing **separate HTML entry artifacts** (and shared chunks) under `front-end/dist/`
  4. WebRTC APIs (`RTCPeerConnection`, `navigator.mediaDevices`) are stubbed in `test-setup.ts` so React tests can import WebRTC-using modules without crashing
**Plans**: 2 plans
Plans:
- [x] 01-01-PLAN.md — Frontend build infrastructure: Vite MPA config, dev/preview `/react` and `/lit` middleware, React shell, WebRTC stubs, build/manifest checks
- [x] 01-02-PLAN.md — Production-style Nginx `/lit` + `/react`, Playwright on `vite preview`, dev topology in `start-frontend.sh`
**UI hint**: yes

### Phase 2: Core Hook Infrastructure
**Goal**: React components can make gRPC calls, receive toast notifications, retrieve filters, and observe backend health through a set of reusable hooks and context providers
**Depends on**: Phase 1
**Requirements**: HOOK-01, HOOK-02, HOOK-03, HOOK-04, HOOK-05
**Success Criteria** (what must be TRUE):
  1. A component wrapped in `ServiceContext` can call `useAsyncGRPC` and receive `{ data, loading, error }` without importing from the DI container
  2. Any component can call `useToast` to display a toast notification visible in the UI
  3. A component using `useFilters` receives a list of available processing filters fetched from the backend
  4. `useHealthMonitor` continuously reflects backend health — status changes within one poll cycle when the backend goes up or down
**Plans**: TBD

### Phase 3: Static Feature UI (Image Processing Path)
**Goal**: Users can upload images, apply filters, view processed results, manage files, view and modify system settings, and see backend health status — all from the React frontend
**Depends on**: Phase 2
**Requirements**: IMG-01, IMG-02, IMG-03, IMG-04, FILE-01, FILE-02, CONF-01, CONF-02, HLTH-01, HLTH-02
**Success Criteria** (what must be TRUE):
  1. User can upload an image, choose one or more filters, trigger processing, and see the processed image rendered in the React frontend
  2. User sees a progress indicator while an image is uploading (not a spinner — actual upload progress)
  3. User can open a file list, select a previously uploaded file, and use it as processing input without re-uploading
  4. User can view current system configuration values and save changes via the settings UI
  5. User sees a health status indicator that changes appearance when the backend becomes unavailable
**Plans**: TBD
**UI hint**: yes

### Phase 4: Video Streaming and WebRTC
**Goal**: Users can start, watch, and stop a real-time filtered video stream in the React frontend with full WebRTC resource cleanup
**Depends on**: Phase 3
**Requirements**: VID-01, VID-02, VID-03, VID-04, VID-05
**Success Criteria** (what must be TRUE):
  1. User can start a video stream and see processed frames rendered on a canvas element at the source frame rate without UI lag
  2. User can switch the video source (camera vs file) before starting a stream
  3. User can stop the stream and the camera indicator light turns off, confirming all WebRTC resources were released
  4. User receives an error toast notification if the WebRTC connection fails or drops mid-stream
**Plans**: TBD
**UI hint**: yes

### Phase 5: Polish and Parity Validation
**Goal**: Developer can verify that the React and Lit frontends produce functionally identical outputs for all features, with no visual or behavioral regressions
**Depends on**: Phase 4
**Requirements**: PAR-01
**Success Criteria** (what must be TRUE):
  1. Developer can open `/react` and `/lit` side-by-side and exercise every feature (image upload, video stream, file list, settings, health) with identical results
  2. The React production build completes with zero TypeScript strict-mode errors and no console errors on page load
  3. CSS rendering of the React frontend matches the Lit frontend with no layout gaps or missing styles
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Scaffold and Infrastructure | 0/2 | Planned | - |
| 2. Core Hook Infrastructure | 0/? | Not started | - |
| 3. Static Feature UI | 0/? | Not started | - |
| 4. Video Streaming and WebRTC | 0/? | Not started | - |
| 5. Polish and Parity Validation | 0/? | Not started | - |
