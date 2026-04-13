# Requirements: CUDA Learning Platform — React Frontend Migration

**Defined:** 2026-04-12
**Core Value:** Practice React best practices while delivering a production-ready frontend with full feature parity to the existing Lit frontend

## v1.0 Requirements

### Scaffold

- [ ] **SCAF-01**: On the **production** user-facing host (Traefik → `web-frontend` / Nginx), `/react` and `/lit` resolve to the correct React and Lit shells on the **same origin** as `/api` and `/ws` (proxied to Go). **Local development** uses a documented split: Vite HTTPS on port 3000 for UI (including `/react` and `/lit` via Vite middleware), Go TLS on port 8443 for API/WebSocket/Connect (`VITE_API_ORIGIN`); Go does not serve the MPA HTML.
- [ ] **SCAF-02**: Vite is configured as a multi-page app (MPA) with separate entry points for Lit and React builds

### Core Hooks

- [ ] **HOOK-01**: Application exposes a singleton ConnectRPC client via React Context (no direct DI container access in components)
- [ ] **HOOK-02**: Components can invoke gRPC calls via `useAsyncGRPC` hook that returns `{ data, loading, error }` state
- [ ] **HOOK-03**: User sees toast notifications for errors and status updates via `ToastContext`
- [ ] **HOOK-04**: User can retrieve and select from available processing filters via `useFilters` hook
- [ ] **HOOK-05**: Application continuously polls backend health and exposes status via `useHealthMonitor` hook

### Image Processing

- [ ] **IMG-01**: User can upload an image file for processing via the React frontend
- [ ] **IMG-02**: User can select one or more filters to apply to the uploaded image
- [ ] **IMG-03**: User can view the processed image result in the React frontend
- [ ] **IMG-04**: User sees upload progress while an image is being uploaded

### File Management

- [ ] **FILE-01**: User can view a list of previously uploaded files in the React frontend
- [ ] **FILE-02**: User can select a file from the list to use as processing input

### Settings

- [ ] **CONF-01**: User can view current system configuration in the React frontend
- [ ] **CONF-02**: User can modify system configuration settings via the React frontend

### Health Monitoring

- [ ] **HLTH-01**: User can see backend health status in the React frontend
- [ ] **HLTH-02**: User receives a visual indicator when the backend is unavailable

### Video Streaming (WebRTC)

- [ ] **VID-01**: User can start a real-time video stream with filter processing in the React frontend
- [ ] **VID-02**: User sees processed video frames displayed via canvas (not React state) at full frame rate
- [ ] **VID-03**: User can select the video source (camera/file) in the React frontend
- [ ] **VID-04**: User can stop the video stream and the application cleans up all WebRTC resources
- [ ] **VID-05**: User receives an error notification if the WebRTC connection fails or drops

### Parity Validation

- [ ] **PAR-01**: Developer can manually verify that `/react` and `/lit` routes produce functionally identical outputs for all features

## Future Requirements (v1.1+)

### Testing Coverage

- **TEST-01**: All React components have React Testing Library unit tests at 80%+ coverage
- **TEST-02**: Existing Vitest frontend tests are migrated to React Testing Library
- **TEST-03**: Production build for both Lit and React entries completes with zero TypeScript strict-mode errors

### Cutover

- **CUT-01**: Lit frontend is removed and Go server serves only the React frontend
- **CUT-02**: All Lit-specific code and routes are cleaned up

## Out of Scope

| Feature | Reason |
|---------|--------|
| Go/C++ backend changes | Zero backend modifications — reuse existing gRPC APIs |
| New features beyond Lit parity | Learning-focused migration; no new capabilities |
| Authentication/authorization | Not in existing system |
| Mobile / React Native | Web-only |
| Zustand state management | Context is the learning goal; add only if Context proves insufficient |
| Redux | Unjustified boilerplate for a single-user app |
| Next.js / SSR | Vite SPA is the established stack |
| RTL test migration | Deferred to v1.1 — parity first, full test suite second |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SCAF-01 | Phase 1 | Pending |
| SCAF-02 | Phase 1 | Pending |
| HOOK-01 | Phase 2 | Pending |
| HOOK-02 | Phase 2 | Pending |
| HOOK-03 | Phase 2 | Pending |
| HOOK-04 | Phase 2 | Pending |
| HOOK-05 | Phase 2 | Pending |
| IMG-01 | Phase 3 | Pending |
| IMG-02 | Phase 3 | Pending |
| IMG-03 | Phase 3 | Pending |
| IMG-04 | Phase 3 | Pending |
| FILE-01 | Phase 3 | Pending |
| FILE-02 | Phase 3 | Pending |
| CONF-01 | Phase 3 | Pending |
| CONF-02 | Phase 3 | Pending |
| HLTH-01 | Phase 3 | Pending |
| HLTH-02 | Phase 3 | Pending |
| VID-01 | Phase 4 | Pending |
| VID-02 | Phase 4 | Pending |
| VID-03 | Phase 4 | Pending |
| VID-04 | Phase 4 | Pending |
| VID-05 | Phase 4 | Pending |
| PAR-01 | Phase 5 | Pending |

**Coverage:**
- v1.0 requirements: 23 total
- Mapped to phases: 23
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-12*
*Last updated: 2026-04-12 after milestone v1.0 scoping*
