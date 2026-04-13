---
phase: 02-core-hook-infrastructure
reviewed: 2026-04-12T12:00:00Z
depth: standard
files_reviewed: 22
files_reviewed_list:
  - front-end/src/infrastructure/grpc/tracing-interceptor.ts
  - front-end/src/infrastructure/grpc/create-grpc-transport.ts
  - front-end/src/application/services/config-service.ts
  - front-end/src/components/app/grpc-status-modal.ts
  - front-end/src/infrastructure/external/remote-management-service.ts
  - front-end/src/react/context/service-context.tsx
  - front-end/src/react/providers/grpc-clients-provider.tsx
  - front-end/src/react/hooks/useAsyncGRPC.ts
  - front-end/src/react/context/toast-context.tsx
  - front-end/src/react/hooks/useToast.ts
  - front-end/src/react/main.tsx
  - front-end/react.html
  - front-end/src/react/test-utils/render-with-service.tsx
  - front-end/src/react/service-context.test.tsx
  - front-end/src/react/use-async-grpc.test.tsx
  - front-end/src/react/use-toast.test.tsx
  - front-end/vitest.config.ts
  - front-end/src/test-setup.ts
  - front-end/src/react/hooks/useFilters.ts
  - front-end/src/react/hooks/useHealthMonitor.ts
  - front-end/src/react/use-filters.test.tsx
  - front-end/src/react/use-health-monitor.test.tsx
findings:
  critical: 0
  warning: 1
  info: 5
  total: 6
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-04-12T12:00:00Z  
**Depth:** standard  
**Files Reviewed:** 22  
**Status:** issues_found

## Summary

Review focused on gRPC/Connect wiring, React context and hooks (`useAsyncGRPC`, `useFilters`, `useHealthMonitor`, toast bridge), Lit `grpc-status-modal`, and matching tests/config. Overall patterns (stable `refetch` callbacks, generation guards for in-flight RPCs, `ConnectError` mapping, visibility-aware polling in `useHealthMonitor`) are sound. The main correctness gap is **Jetson monitor stream lifecycle** in `grpc-status-modal`: teardown does not stop the server-streaming consumer, and the local `AbortController` is not wired into the Connect call, so abort does not cancel the stream.

No critical security issues were found (Lit text bindings avoid raw HTML injection for terminal lines). Test coverage for hooks is reasonable; gaps are noted under Info (e.g. `useAsyncGRPC` dependency-array behavior, toast bridge when host is missing).

## Warnings

### WR-01: Jetson monitor stream not cancelled on modal teardown

**File:** `front-end/src/components/app/grpc-status-modal.ts:350-358`, `551-580`  
**File:** `front-end/src/infrastructure/external/remote-management-service.ts:74-98`

**Issue:** `disconnectedCallback` clears the accelerator health interval and removes listeners but **does not** call `stopMonitoring()` (or otherwise tear down the `monitorJetsonNano` async loop). Separately, `startMonitoring` creates `monitorStreamAbortController` and `stopMonitoring()` calls `abort()`, but **`monitorJetsonNano` never receives that `AbortSignal`**. In Connect, server-streaming methods accept call options including `signal`; without it, `abort()` on a standalone controller has no effect on the active stream, and the `for await` loop in `RemoteManagementService.monitorJetsonNano` can keep running after the modal is closed or removed.

**Fix:**

1. In `disconnectedCallback`, invoke full teardown (e.g. `this.close()` or at minimum `stopMonitoring()` plus any interval cleanup you already do).
2. Extend `remoteManagementService.monitorJetsonNano` to accept an optional `AbortSignal` (or options object) and pass it through:

```typescript
// remote-management-service.ts (illustrative)
async monitorJetsonNano(
  onData: (data: string) => void,
  onError?: (error: Error) => void,
  signal?: AbortSignal
): Promise<void> {
  const request = new MonitorJetsonNanoRequest({});
  const stream = this.client.monitorJetsonNano(request, { signal });
  for await (const response of stream) {
    // ...
  }
}
```

3. In `grpc-status-modal`, pass `this.monitorStreamAbortController.signal` into `monitorJetsonNano` so `stopMonitoring()` actually cancels the RPC.

## Info

### IN-01: Unused imports and dead import in `grpc-status-modal`

**File:** `front-end/src/components/app/grpc-status-modal.ts:7-16`

**Issue:** `RemoteManagementService as RemoteManagementServiceClient`, `StartJetsonNanoStatus`, and `processorCapabilitiesService` appear unused. This obscures real dependencies and can hide unused capability wiring.

**Fix:** Remove unused imports or use them if intended features were never connected.

### IN-02: `addTerminalLine` severity parameter is unused

**File:** `front-end/src/components/app/grpc-status-modal.ts:503-515`

**Issue:** The second parameter `type` is never stored; rendering re-derives styling from substring heuristics on the line text. Call sites pass `'info' | 'success' | 'error'` but that information is dropped.

**Fix:** Either encode type in stored entries (e.g. structured objects) or remove the parameter and rely on one classification path.

### IN-03: `test-setup` logger mock path likely does not match app imports

**File:** `front-end/src/test-setup.ts:5-15`

**Issue:** `vi.mock('./services/otel-logger', ...)` resolves under `src/services/`, while production code imports `@/infrastructure/observability/otel-logger`. The mock may not apply to modules using the alias, so tests might still hit the real logger or miss intended stubs.

**Fix:** Mock the same module specifier the app uses, e.g. `vi.mock('@/infrastructure/observability/otel-logger', ...)`, or a path Vitest resolves to the same module graph entry.

### IN-04: `useAsyncGRPC` — spreading `deps` into `useEffect` dependencies

**File:** `front-end/src/react/hooks/useAsyncGRPC.ts:61-66`

**Issue:** Callers must pass a stable `deps` array. Inline arrays/objects in `deps` will retrigger the effect every render (risk of repeated fetches). Not a bug with correct usage; worth documenting in a one-line JSDoc or tightening the type/docs.

**Fix:** Document expected stability, or accept a single `deps` array only (no rest spread) so callers are forced to memoize explicitly.

### IN-05: Unused protobuf imports in `remote-management-service`

**File:** `front-end/src/infrastructure/external/remote-management-service.ts:4-11`

**Issue:** Several symbols from `remote_management_service_pb` are imported but unused (`StartJetsonNanoResponse`, `AcceleratorHealthStatus`, `MonitorJetsonNanoResponse` per current file).

**Fix:** Remove unused imports to satisfy linters and reduce noise.

## Test gaps (brief)

- **`useAsyncGRPC`:** No test that stale completions are ignored when `deps` change mid-flight (generation guard is implemented; an explicit test would lock the behavior).
- **`ToastProvider`:** No test for the “no `toast-container`” path (warn + no-op); behavior is implemented in `toast-context.tsx`.

---

_Reviewed: 2026-04-12T12:00:00Z_  
_Reviewer: Claude (gsd-code-reviewer)_  
_Depth: standard_
