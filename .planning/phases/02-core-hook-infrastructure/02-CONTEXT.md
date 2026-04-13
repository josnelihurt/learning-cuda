# Phase 2: Core Hook Infrastructure - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

React components on `/react` gain **shared infrastructure only**: ConnectRPC access through React Context (no Lit DI container), a reusable async gRPC hook pattern, toast notifications, a filters hook backed by the real ListFilters RPC, and a health monitor hook aligned with existing accelerator health checks. **No** full feature UI (that is Phase 3); hooks may be proven with minimal demo usage in the shell if the planner requires it.

**Out of scope:** New product features, Lit refactors, backend changes, WebRTC streaming hooks (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Connect transport and clients (HOOK-01, HOOK-02)

- **D-01:** Use **one** shared `createConnectTransport` per React app tree (`baseUrl: window.location.origin`, same origin pattern as Lit `ConfigService` / other services) so `/grpc` and cookies behave like today under Vite and Nginx.
- **D-02:** Register **telemetry/trace interceptors** on that transport consistent with Lit — reuse or share the same interceptor logic as `front-end/src/application/services/config-service.ts` (`tracingInterceptor` + `telemetryService.getTraceHeaders()`), not a stripped transport.
- **D-03:** Expose **typed `PromiseClient` instances** (or a small facade) via React Context — components and hooks obtain clients from context; **no** per-call `createConnectTransport` inside random components.
- **D-04:** Do **not** expose the Lit `Container` / DI to React; Context providers are mounted from `front-end/src/react/main.tsx` (or a dedicated `providers.tsx`).

### Toasts (HOOK-03)

- **D-05:** **Bridge** to the existing Lit `<toast-container>` custom element: resolve `document.querySelector('toast-container')` and call `.success`, `.error`, `.warning`, `.info` with the same signatures as `ToastContainer` in `toast-container.ts` (mirror `filter-panel.ts` / `sync-flags-button.ts` pattern).
- **D-06:** Ensure the React MPA entry includes a **`toast-container` in the DOM** — `react.html` currently has only `#root`; add `<toast-container></toast-container>` (or equivalent host) so the bridge works on `/react` without depending on Lit layout.
- **D-07:** Implement **`useToast`** (and optionally `ToastContext`) as the stable React API; implementation delegates to the web component bridge.

### `useAsyncGRPC` shape (HOOK-02)

- **D-08:** Beyond roadmap `{ data, loading, error }`, expose **`refetch`** (or `execute`) so components can re-invoke the RPC without remounting; supports reads and simple mutations until Phase 3 patterns evolve.
- **D-09:** Map Connect errors to a stable **`error`** shape (message / code) for UI; planner chooses details.

### `useFilters` (HOOK-04)

- **D-10:** Load filters via **ImageProcessorService** `listFilters` (generated client from `front-end/src/gen/`), using the shared transport from D-01–D-03.
- **D-11:** Expose **list + loading + error + refetch**; selection state can stay local to consumers in Phase 3 unless the planner specifies a shared store in Phase 2.

### `useHealthMonitor` (HOOK-05)

- **D-12:** Align semantics with existing Lit behavior: **`RemoteManagementService.checkAcceleratorHealth`** and `AcceleratorHealthStatus` (see `grpc-status-modal.ts` usage), not an ad-hoc health definition.
- **D-13:** **Pause polling** when `document.visibilityState === 'hidden'`; resume when visible to avoid useless traffic in background tabs.
- **D-14:** Default poll interval is **planner/implementation choice** — start around **15–30 seconds** unless requirements or UX demand otherwise; must meet roadmap “within one poll cycle” when backend flips.

### Claude's Discretion

- Exact Context provider file split (`providers.tsx` vs inline in `main.tsx`).
- Whether `useAsyncGRPC` uses manual `useEffect` state vs a minimal internal helper; testing strategy (Vitest + happy-dom).
- Exact TypeScript types for generic service methods on the hook.
- Whether to add `AbortSignal` / strict cancellation on unmount for in-flight RPCs.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap

- `.planning/REQUIREMENTS.md` — HOOK-01 through HOOK-05
- `.planning/ROADMAP.md` — Phase 2 goal and success criteria
- `.planning/phases/01-scaffold-and-infrastructure/01-CONTEXT.md` — React layout, no Lit DI, shared `front-end/src` with Lit

### ConnectRPC and generated clients

- `front-end/src/application/services/config-service.ts` — reference `createConnectTransport` + `tracingInterceptor` + `createPromiseClient` pattern
- `front-end/src/gen/image_processor_service_connect.ts` — `ListFilters` RPC for HOOK-04
- `front-end/src/gen/remote_management_service_connect.ts` — `CheckAcceleratorHealth` for HOOK-05

### Lit reference implementations (parity, not imports from React into Lit)

- `front-end/src/components/app/toast-container.ts` — toast API surface for the bridge
- `front-end/src/components/app/grpc-status-modal.ts` — accelerator health loading and status interpretation
- `front-end/src/components/app/filter-panel.ts` — example of `getToastManager()` / toast usage

### React entry

- `front-end/src/react/main.tsx` — provider mount point
- `front-end/react.html` — must host `toast-container` per D-06

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable assets

- **Connect + interceptors:** Lit `config-service.ts` shows `baseUrl: window.location.origin`, `useHttpGet: true`, and OpenTelemetry header propagation.
- **Toast UI:** `ToastContainer` custom element with `success` / `error` / `warning` / `info`.
- **Health:** `grpc-status-modal.ts` calls `remoteManagementService.checkAcceleratorHealth()` and maps `AcceleratorHealthStatus`.

### Established patterns

- Lit code uses **DOM query** for `toast-container` where needed; React will centralize that behind `useToast`.
- Multiple Lit services each construct transport — React phase **consolidates** to one transport to avoid drift.

### Integration points

- Wrap React tree in `main.tsx` with providers that depend on a single transport.
- `react.html` must include **both** `#root` and **`toast-container`** for HOOK-03.

</code_context>

<specifics>
## Specific Ideas

- User chose to discuss **all four** gray areas in discuss-phase; decisions above follow the recommended options (shared transport + context, toast bridge + ensure element on React page, extended async hook surface, health polling with visibility pause and alignment to `CheckAcceleratorHealth`).

</specifics>

<deferred>
## Deferred Ideas

- **React-only toast component** — deferred to parity / polish if bridging becomes painful (Phase 5 or backlog).
- **TanStack Query** as the sole data layer — not required for Phase 2; revisit if hooks become too complex.

### Reviewed Todos (not folded)

- None (todo match-phase returned no items).

</deferred>

---
*Phase: 02-core-hook-infrastructure*
*Context gathered: 2026-04-13*
