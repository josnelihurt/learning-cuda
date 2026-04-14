# React Full Parity Migration Tasks

## Goal
Replace remaining Lit-based UI runtime pieces on the React route with native React implementations, while preserving behavior, UX, and test coverage.

## Current Reality (What is still Lit on `/react`)
- `video-grid` is a Lit custom element imported in `src/front-end/src/react/main.tsx`.
- `camera-preview` is a Lit custom element imported in `src/front-end/src/react/main.tsx`.
- Several cross-cutting UI elements still rely on Lit custom elements: `toast-container`, `stats-panel`, `source-drawer`, `add-source-fab`, `image-selector-modal`, `grpc-status-modal`, `tools-dropdown`, `feature-flags-*`, `app-tour`, `information-banner`.
- React currently hosts these elements and wires them via imperative refs/listeners in `VideoGridHost.tsx`, instead of owning the rendering tree end-to-end.

## Migration Scope (Tasks by Priority)

### P0 - Core video surface parity
- [ ] Build `ReactVideoGrid` to replace Lit `video-grid` rendering and internal source state.
- [ ] Build `ReactVideoSourceCard` to replace Lit `video-source-card` (select, close, change image actions).
- [ ] Build `ReactCameraPreview` to replace Lit `camera-preview` capture/preview behavior.
- [ ] Move per-source pipeline state (`filters`, `resolution`, `currentImageSrc`, `originalImageSrc`, streaming metadata) into React state/store.
- [ ] Replace imperative `grid.*` calls from `VideoGridHost` with typed React callbacks/handlers.

### P0 - Streaming and processing behavior parity
- [ ] Preserve source lifecycle parity: add source, remove source, auto-select first source, selected-source switching.
- [ ] Preserve static image processing parity: resolution changes + filter application + image replacement.
- [ ] Preserve video source processing parity: restart/send-start behavior when filters change.
- [ ] Preserve camera path parity: frame capture loop, filter transport, and source image refresh.
- [ ] Keep WebRTC session lifecycle parity (create, heartbeat, stop/close on remove/unload).

### P1 - Replace Lit overlays and controls that block full React ownership
- [ ] Replace `source-drawer` with React component and keep source/video upload/select workflows.
- [ ] Replace `image-selector-modal` with React modal and keep `change image` flow parity.
- [ ] Replace `add-source-fab` and `accelerator-status-fab` with React components.
- [ ] Replace `stats-panel` and `toast-container` bridges with React-native equivalents or adapters.

### P1 - Sidebar/control orchestration cleanup
- [ ] Remove DOM event bridge logic (`CustomEvent` wiring) once components are React-native.
- [ ] Convert remaining orchestration from document queries to provider/hooks state flow.
- [ ] Ensure filter panel, selected source, accelerator, and resolution state stay fully React-controlled.

### P2 - Secondary Lit widgets and route cleanup
- [ ] Decide migration strategy for `tools-dropdown`, `feature-flags-button/modal`, `sync-flags-button`, `grpc-status-modal`, `app-tour`, `information-banner`, `version-tooltip-lit`.
- [ ] Remove Lit component imports from `src/front-end/src/react/main.tsx` after each React replacement is complete.
- [ ] Keep `/` (Lit) and `/react` behavior aligned until cutover decision.

## Validation Checklist (Definition of Done)
- [ ] No Lit custom element imports remain in `src/front-end/src/react/main.tsx` for migrated features.
- [ ] `VideoGridHost` no longer depends on imperative custom-element API calls for migrated areas.
- [ ] React tests cover all migrated flows:
  - [ ] Add/remove source
  - [ ] Select source + sync filters/resolution
  - [ ] Apply filters to image/video/camera sources
  - [ ] Change image flow
  - [ ] Stream start/stop behavior
- [ ] Manual parity checks pass against Lit route for the same scenarios.
- [ ] Existing React test suite remains green.

## Suggested Execution Order
1. `ReactVideoSourceCard`
2. `ReactCameraPreview`
3. `ReactVideoGrid`
4. Replace image-change modal + source drawer in React
5. Remove `video-grid` and `camera-preview` Lit imports from React entry
6. Migrate remaining secondary widgets

## Risk Notes
- The highest migration risk is behavior drift in the video/camera processing loops and WebRTC/session lifecycle.
- Keep transport/service layer contracts stable during UI migration to reduce regression surface.
- Prefer thin adapters during transition, then delete adapters after full feature parity is verified.
