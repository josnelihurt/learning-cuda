---
phase: 02-core-hook-infrastructure
plan: "01"
subsystem: frontend
tags: [react, connectrpc, grpc, vitest, lit, toast, context]

requires:
  - phase: 01-scaffold-and-infrastructure
    provides: MPA /react entry, Vite + React, shared src tree
provides:
  - Shared tracingInterceptor and createGrpcConnectTransport factory
  - React ServiceContext with ImageProcessor and RemoteManagement PromiseClients
  - useAsyncGRPC with data, loading, error, refetch and abort on unmount
  - ToastProvider and useToast bridging to toast-container custom element
affects:
  - Phase 02 follow-on plans (useFilters, useHealthMonitor)
  - Phase 03 React feature UI

tech-stack:
  added: []
  patterns:
    - "Single Connect transport per React tree via GrpcClientsProvider useMemo"
    - "Lit/React parity for trace headers via shared tracingInterceptor"
    - "DOM bridge for toasts with logger.warn when host element missing"

key-files:
  created:
    - front-end/src/infrastructure/grpc/tracing-interceptor.ts
    - front-end/src/infrastructure/grpc/create-grpc-transport.ts
    - front-end/src/react/context/service-context.tsx
    - front-end/src/react/providers/grpc-clients-provider.tsx
    - front-end/src/react/hooks/useAsyncGRPC.ts
    - front-end/src/react/hooks/useToast.ts
    - front-end/src/react/context/toast-context.tsx
    - front-end/src/react/test-utils/render-with-service.tsx
    - front-end/src/react/service-context.test.tsx
    - front-end/src/react/use-async-grpc.test.tsx
    - front-end/src/react/use-toast.test.tsx
  modified:
    - front-end/src/application/services/config-service.ts
    - front-end/src/components/app/grpc-status-modal.ts
    - front-end/src/infrastructure/external/remote-management-service.ts
    - front-end/src/react/main.tsx
    - front-end/react.html
    - front-end/vitest.config.ts
    - front-end/src/test-setup.ts

key-decisions:
  - "ToastProvider wraps GrpcClientsProvider so toast API stays independent of gRPC context"
  - "useAsyncGRPC maps failures with ConnectError.from for stable { message, code } under Vitest bundling"
  - "Request generation guard in useAsyncGRPC so aborted in-flight work does not strand loading state"

patterns-established:
  - "Import @/components/app/toast-container in React main for custom element registration"
  - "renderWithService helper mounts ServiceContext.Provider with stub clients for hook tests"

requirements-completed: [HOOK-01, HOOK-02, HOOK-03]

duration: 25min
completed: 2026-04-12
---

# Phase 02 Plan 01: Core hook infrastructure (gRPC + toasts) Summary

**Shared Connect tracing interceptor and transport factory, React `GrpcClientsProvider` with `useAsyncGRPC` (refetch + stable errors), and `ToastProvider`/`useToast` bridging to the existing Lit `<toast-container>` on `/react`.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-04-12 (executor session)
- **Completed:** 2026-04-12
- **Tasks:** 3
- **Files touched:** 18 (11 new, 7 updated)

## Accomplishments

- Deduplicated Lit `tracingInterceptor` into `front-end/src/infrastructure/grpc/` and added `createGrpcConnectTransport` for React and future callers.
- React tree receives one shared transport and typed `PromiseClient`s for `ImageProcessorService` and `RemoteManagementService` via context.
- `useAsyncGRPC` exposes `refetch`, aborts on unmount/superseded requests, and normalizes errors to `{ message, code? }`.
- `/react` page includes `<toast-container>`; React entry registers the element and exposes toasts through context-backed `useToast`.

## Task Commits

1. **Task 1: Extract shared tracing interceptor and Connect transport factory** — `0232d58` (feat)
2. **Task 2: ServiceContext, GrpcClientsProvider, useAsyncGRPC, mount in main.tsx** — `2aa8b57` (feat)
3. **Task 3: Toast DOM host, ToastContext, useToast, Vitest HOOK-03** — `be8f3ee` (feat)

## Files Created/Modified

- `front-end/src/infrastructure/grpc/tracing-interceptor.ts` — Shared OpenTelemetry trace header interceptor for Connect.
- `front-end/src/infrastructure/grpc/create-grpc-transport.ts` — Same-origin transport with interceptor and `useHttpGet: true`.
- `front-end/src/react/context/service-context.tsx` — `GrpcClients` type, `ServiceContext`, `useServiceContext`.
- `front-end/src/react/providers/grpc-clients-provider.tsx` — Single transport and both Promise clients in context.
- `front-end/src/react/hooks/useAsyncGRPC.ts` — Async RPC executor hook with refetch and cancellation.
- `front-end/src/react/context/toast-context.tsx` — `ToastProvider` and DOM bridge to `ToastContainer`.
- `front-end/src/react/hooks/useToast.ts` — Context consumer with guard when provider is missing.
- `front-end/src/react/test-utils/render-with-service.tsx` — Minimal `createRoot` + `act` test helper.
- `front-end/src/react/*.test.tsx` — Vitest coverage for HOOK-01–03.
- `front-end/vitest.config.ts` — `@vitejs/plugin-react` for TSX under Vitest.
- `front-end/src/test-setup.ts` — `IS_REACT_ACT_ENVIRONMENT` for React `act` warnings.
- `front-end/react.html` — Host for `<toast-container>`.
- `front-end/src/react/main.tsx` — Side-effect toast import, `ToastProvider`, `GrpcClientsProvider`.
- Lit call sites updated to import shared `tracingInterceptor` (`config-service`, `grpc-status-modal`, `remote-management-service`).

## Decisions Made

- `ToastProvider` sits outside `GrpcClientsProvider` so toast usage does not depend on gRPC wiring.
- Error mapping uses `ConnectError.from(e)` so caught values always yield a consistent `message`/`code` string pair (including under Vitest transforms).
- `remote-management-service` imports the shared interceptor via `@/infrastructure/grpc/tracing-interceptor` so acceptance checks and bundler resolution stay aligned with the rest of `src/`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Vitest did not transform React TSX until the React plugin was enabled**

- **Found during:** Task 2 (first `npx vitest run` on new `.tsx` tests)
- **Issue:** JSX tests failed with `React is not defined`.
- **Fix:** Registered `@vitejs/plugin-react` in `vitest.config.ts` and set `globalThis.IS_REACT_ACT_ENVIRONMENT = true` in `src/test-setup.ts`.
- **Files modified:** `front-end/vitest.config.ts`, `front-end/src/test-setup.ts`
- **Verification:** `npx vitest run src/react/service-context.test.tsx src/react/use-async-grpc.test.tsx`
- **Committed in:** `2aa8b57`

**2. [Rule 1 - Bug] useAsyncGRPC could leave loading stuck and miss errors when requests were aborted or `instanceof` split across bundles**

- **Found during:** Task 2 (`maps ConnectError` test timeout / empty error capture)
- **Issue:** Strict cleanup and/or Vitest module boundaries meant `instanceof ConnectError` and `signal.aborted` handling did not match runtime behavior; push-during-render + `act`+`waitFor` also hid updates.
- **Fix:** Added a monotonic request generation guard so only the latest request clears loading; mapped errors with `ConnectError.from`; adjusted the error test to observe state in `useEffect` and `vi.waitFor` outside a conflicting `act` wrapper.
- **Files modified:** `front-end/src/react/hooks/useAsyncGRPC.ts`, `front-end/src/react/use-async-grpc.test.tsx`
- **Verification:** Vitest suite for `use-async-grpc.test.tsx`
- **Committed in:** `2aa8b57`

**3. [Rule 3 - Blocking] Acceptance `rg` for three Lit import paths required a string match on `infrastructure/grpc/tracing-interceptor`**

- **Found during:** Task 1 (acceptance criteria grep)
- **Issue:** Relative `../grpc/tracing-interceptor` did not match the plan’s `rg` pattern.
- **Fix:** Switched `remote-management-service` to `@/infrastructure/grpc/tracing-interceptor`.
- **Files modified:** `front-end/src/infrastructure/external/remote-management-service.ts`
- **Verification:** Plan `rg` for three import lines
- **Committed in:** `0232d58`

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 bug)

**Impact on plan:** Scoped to test harness and robustness of the hook; no change to trust boundaries (`baseUrl` remains `window.location.origin` only; toast text still flows through Lit templates).

## Issues Encountered

- Residual React `act(...)` warnings on one async test path after switching to `useEffect` observation; tests pass and behavior is covered.

## User Setup Required

None — unit tests use happy-dom mocks only.

## Next Phase Readiness

- HOOK-01–03 automated proof exists; ready for HOOK-04/05 plans on this foundation.
- Orchestrator should advance `.planning/STATE.md` / `ROADMAP.md` and requirement checkboxes after merge (not updated in this executor run per instructions).

## Self-Check: PASSED

- Key deliverables exist on disk under `front-end/src/infrastructure/grpc/`, `front-end/src/react/`, and `front-end/react.html`.
- Task commits present on `main`: `0232d58`, `2aa8b57`, `be8f3ee`.
- `cd front-end && npx vitest run src/react/service-context.test.tsx src/react/use-async-grpc.test.tsx src/react/use-toast.test.tsx` exits 0.

---
*Phase: 02-core-hook-infrastructure*
*Completed: 2026-04-12*
