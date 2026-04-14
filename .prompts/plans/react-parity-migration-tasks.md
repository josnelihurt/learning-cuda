# React Parity Remaining Tasks

## Goal
Finish the remaining work required to reach 100% functional and UX parity between `/` (Lit) and `/react` (React-native route), with no Lit runtime widgets left on `/react`.

## What Is Already Migrated
- React-native video surface is in place: `ReactVideoGrid`, `ReactVideoSourceCard`, `ReactCameraPreview`.
- Source lifecycle and processing flows are React-owned in `VideoGridHost`:
  - add/remove/select source
  - image filter application + resolution changes
  - video restart on filter changes
  - camera frame loop + frame transport
  - WebRTC create/close hooks
- React-native route overlays implemented for migrated scope:
  - source drawer
  - image selector modal
  - add source FAB
  - accelerator status FAB
  - stats panel
  - toast rendering
- Lit imports for migrated widgets were removed from `src/front-end/src/react/main.tsx`.

## Remaining Work To Reach 100%

### 1) Migrate remaining Lit-only widgets used on `/react`
- [ ] Replace `tools-dropdown` with React component.
- [ ] Replace `feature-flags-button` with React component.
- [ ] Replace `feature-flags-modal` with React component.
- [ ] Replace `sync-flags-button` with React component (or remove if obsolete by design).
- [ ] Replace `grpc-status-modal` with React component.
- [ ] Replace `app-tour` with React component.
- [ ] Replace `information-banner` with React component.
- [ ] Replace `version-tooltip-lit` with React component.

### 2) Remove remaining Lit dependencies from React route entry
- [ ] Remove all remaining Lit component imports from `src/front-end/src/react/main.tsx`.
- [ ] Remove remaining Lit host tags from `src/front-end/react.html` for widgets migrated to React.
- [ ] Ensure `/react` boots and runs with React-owned UI tree only.

### 3) Close remaining behavior parity gaps
- [ ] Verify camera behavior matches Lit route under long-running sessions (no freeze, no permission loop, continuous processing).
- [ ] Verify source numbering/selection parity under add/remove stress scenarios.
- [ ] Verify WebRTC heartbeat/session teardown parity on source remove and tab unload.
- [ ] Verify upload/select workflows parity (image + video) if upload widgets are expected inside React drawer.

### 4) Testing and validation to declare parity complete
- [ ] Add/expand React tests to cover migrated and remaining parity-critical flows:
  - [ ] add/remove source
  - [ ] selected source synchronization with filter panel and resolution
  - [ ] static image processing updates
  - [ ] video restart semantics on filter change
  - [ ] camera continuous frame processing
  - [ ] change image flow
  - [ ] stream start/stop and cleanup behavior
- [ ] Run React suite green (`npm run test` in `src/front-end`).
- [ ] Run production build green (`npm run build` in `src/front-end`).
- [ ] Execute manual parity checklist against `/` route for same scenarios and confirm no UX/behavior drift.

## Definition of Done (100% Parity)
- [ ] `/react` has no Lit custom element imports for runtime UI.
- [ ] `/react` has no Lit host elements required in `react.html` for runtime UI.
- [ ] All dashboard interactions on `/react` are controlled by React state/hooks/providers.
- [ ] Automated tests cover parity-critical workflows and pass.
- [ ] Manual parity checks against Lit route pass for image, video, camera, and control-plane features.
