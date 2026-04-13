# Phase 02: Core Hook Infrastructure — Research

**Researched:** 2026-04-12  
**Domain:** ConnectRPC (`@connectrpc/connect-web`) + React 18/19 + Vitest (happy-dom)  
**Overall confidence:** HIGH for transport/toast/RPC names (codebase-verified); MEDIUM for telemetry parity on `/react` until an explicit bootstrap step exists

<user_constraints>

## User Constraints (from 02-CONTEXT.md)

### Locked Decisions

**Connect transport and clients (HOOK-01, HOOK-02)**

- **D-01:** One shared `createConnectTransport` per React app tree (`baseUrl: window.location.origin`, same-origin pattern as Lit `ConfigService` / other services) so `/grpc` and cookies behave like today under Vite and Nginx.
- **D-02:** Register telemetry/trace interceptors on that transport consistent with Lit — reuse or share the same interceptor logic as `front-end/src/application/services/config-service.ts` (`tracingInterceptor` + `telemetryService.getTraceHeaders()`), not a stripped transport.
- **D-03:** Expose typed `PromiseClient` instances (or a small facade) via React Context — components and hooks obtain clients from context; no per-call `createConnectTransport` inside random components.
- **D-04:** Do not expose the Lit `Container` / DI to React; Context providers are mounted from `front-end/src/react/main.tsx` (or a dedicated `providers.tsx`).

**Toasts (HOOK-03)**

- **D-05:** Bridge to the existing Lit `<toast-container>` custom element: resolve `document.querySelector('toast-container')` and call `.success`, `.error`, `.warning`, `.info` with the same signatures as `ToastContainer` in `toast-container.ts` (mirror `filter-panel.ts` / `sync-flags-button.ts` pattern).
- **D-06:** Ensure the React MPA entry includes a `toast-container` in the DOM — `react.html` currently has only `#root`; add `<toast-container></toast-container>` so the bridge works on `/react` without depending on Lit layout.
- **D-07:** Implement `useToast` (and optionally `ToastContext`) as the stable React API; implementation delegates to the web component bridge.

**`useAsyncGRPC` shape (HOOK-02)**

- **D-08:** Beyond `{ data, loading, error }`, expose `refetch` (or `execute`) so components can re-invoke the RPC without remounting.
- **D-09:** Map Connect errors to a stable `error` shape (message / code) for UI; planner chooses details.

**`useFilters` (HOOK-04)**

- **D-10:** Load filters via **ImageProcessorService** `listFilters` (generated client from `front-end/src/gen/`), using the shared transport from D-01–D-03.
- **D-11:** Expose list + loading + error + refetch; selection state stays local to consumers in Phase 3 unless the planner specifies a shared store in Phase 2.

**`useHealthMonitor` (HOOK-05)**

- **D-12:** Align semantics with existing Lit behavior: `RemoteManagementService.checkAcceleratorHealth` and `AcceleratorHealthStatus` (see `grpc-status-modal.ts` usage), not an ad-hoc health definition.
- **D-13:** Pause polling when `document.visibilityState === 'hidden'`; resume when visible.
- **D-14:** Default poll interval ~**15–30 seconds** (planner/implementation choice); must meet roadmap “within one poll cycle” when backend flips.

### Claude's Discretion

- Exact Context provider file split (`providers.tsx` vs inline in `main.tsx`).
- Whether `useAsyncGRPC` uses manual `useEffect` state vs a minimal internal helper; testing strategy (Vitest + happy-dom).
- Exact TypeScript types for generic service methods on the hook.
- Whether to add `AbortSignal` / strict cancellation on unmount for in-flight RPCs.

### Deferred Ideas (OUT OF SCOPE)

- React-only toast component — deferred to parity / polish if bridging becomes painful (Phase 5 or backlog).
- TanStack Query as the sole data layer — not required for Phase 2; revisit if hooks become too complex.

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research support |
|----|-------------|------------------|
| HOOK-01 | Application exposes a singleton ConnectRPC client via React Context (no direct DI container access in components) | Single `createConnectTransport` + `createPromiseClient` for each service; provider in `main.tsx` / `providers.tsx` (D-01–D-04). |
| HOOK-02 | Components invoke gRPC via `useAsyncGRPC` returning `{ data, loading, error }` | Unary `PromiseClient` methods + local state; optional second-arg call options include `signal` in Connect ES 1.7 [VERIFIED: `front-end/node_modules/@connectrpc/connect/dist/esm/promise-client.js`]; extend with `refetch` (D-08, D-09). |
| HOOK-03 | Toasts via `ToastContext` | Bridge to `ToastContainer` API in `toast-container.ts`; DOM host in `react.html` + side-effect import to register the custom element (D-05–D-07). |
| HOOK-04 | Retrieve/select filters via `useFilters` | `ImageProcessorService.listFilters` in `front-end/src/gen/image_processor_service_connect.ts` + `ListFiltersRequest` / `ListFiltersResponse` in `image_processor_service_pb.ts` (D-10, D-11). |
| HOOK-05 | Poll backend health via `useHealthMonitor` | `RemoteManagementService.checkAcceleratorHealth` in `front-end/src/gen/remote_management_service_connect.ts`; health enum `AcceleratorHealthStatus` and `CheckAcceleratorHealthResponse` in `remote_management_service_pb.ts`; parity with `grpc-status-modal.ts` `getStatusInfo()` (D-12–D-14). |

</phase_requirements>

## Executive summary

Phase 2 sits entirely on the Phase 1 MPA scaffold: `front-end/react.html` and `front-end/src/react/main.tsx` are the mount surface. The main execution risks are **integration correctness** (one transport, identical interceptor behavior to Lit) and **DOM prerequisites** for toasts (custom element **defined** and **present**). React’s entry does not run `front-end/src/main.ts`, so **OpenTelemetry is not initialized the same way as on `/lit`** unless Phase 2 explicitly adds a bootstrap; interceptors remain safe because `telemetryService.getTraceHeaders()` returns `{}` when telemetry is disabled [VERIFIED: `front-end/src/infrastructure/observability/telemetry-service.ts`].

**Dependencies on Phase 1:** `/react` must keep resolving to the React shell (SCAF-01/02). No backend changes; all RPCs are existing Connect services.

**Ordering recommendation:** (1) Shared transport factory + interceptor deduplication, (2) React context providers and client creation, (3) `useAsyncGRPC`, (4) side-effect import + `<toast-container>` in `react.html` + `useToast`/`ToastContext`, (5) `useFilters`, (6) `useHealthMonitor` with visibility listener and interval cleanup.

**Primary recommendation:** Centralize `createConnectTransport({ baseUrl: window.location.origin, interceptors: [tracingInterceptor], useHttpGet: true })` to match `config-service.ts` and `remote-management-service.ts`, expose `PromiseClient`s from one provider, and implement domain hooks against those clients only.

## Technical approach

### Standard stack (pinned in repo)

| Package | Installed | Purpose |
|---------|-----------|---------|
| `@connectrpc/connect` | 1.7.0 [VERIFIED: `front-end/node_modules/@connectrpc/connect/package.json`] | `createPromiseClient`, `ConnectError`, `Interceptor`, unary call options (`signal`, `timeoutMs`) [VERIFIED: `promise-client.js`] |
| `@connectrpc/connect-web` | ^1.7.0 [VERIFIED: `front-end/package.json`] | `createConnectTransport` for browser |
| `@bufbuild/protobuf` | ^1.10.1 | Request/response message types for `ListFilters`, `CheckAcceleratorHealth` |
| `react` / `react-dom` | ^19.2.5 | Context, hooks, StrictMode (current `main.tsx`) |
| `vitest` + `happy-dom` | ^1.2.0 / ^12.10.3 | Unit tests [VERIFIED: `front-end/vitest.config.ts`] |

Registry note: latest `@connectrpc/connect` on npm is 2.x [VERIFIED: `npm view @connectrpc/connect version` → 2.1.1]; the project intentionally stays on 1.7.x with `createPromiseClient` — do not bump major without an approved migration ([user rule: no standard/version changes without approval]).

### ConnectRPC + React 18/19 patterns

- **Single transport:** Instantiate one transport, then `createPromiseClient(ImageProcessorService, transport)` and `createPromiseClient(RemoteManagementService, transport)` (same pattern as `streamConfigService` in `config-service.ts` and `RemoteManagementService` in `front-end/src/infrastructure/external/remote-management-service.ts`, but **one** shared transport for both).
- **Interceptors:** Reuse the same header injection as Lit:

```10:16:front-end/src/application/services/config-service.ts
const tracingInterceptor: Interceptor = (next) => async (req) => {
  const headers = telemetryService.getTraceHeaders();
  for (const [key, value] of Object.entries(headers)) {
    req.header.set(key, value);
  }
  return await next(req);
};
```

- **Context provider layout:** Wrap `<App />` in `main.tsx` with a provider that supplies an object like `{ imageProcessorClient, remoteManagementClient }` or a named `GrpcClientsContext`. Roadmap wording references `ServiceContext` — name is implementation detail as long as hooks read clients only from context (HOOK-01).
- **File placement under `front-end/src/react/`:** e.g. `providers.tsx` (or `grpc-provider.tsx`), `hooks/useAsyncGRPC.ts`, `hooks/useToast.ts`, `hooks/useFilters.ts`, `hooks/useHealthMonitor.ts`, `context/grpc-clients-context.tsx`. Keep imports reachable from `@/` alias (`vitest.config.ts` maps `@` → `./src`).

### Interceptor deduplication (planner task)

The same `tracingInterceptor` closure is duplicated today in `config-service.ts`, `grpc-status-modal.ts`, and `remote-management-service.ts`. Phase 2 should **extract** a tiny shared helper (e.g. under `front-end/src/infrastructure/grpc/` or next to observability) so React and Lit cannot drift — still one physical interceptor implementation used by the React transport factory.

## Hook design notes

### `useAsyncGRPC`

- **Shape:** `{ data, loading, error, refetch }` per D-08; initial fetch on mount vs manual `execute` is discretionary.
- **Cancellation:** Connect unary functions accept `(input, options)` where `options.signal` is forwarded to `transport.unary` [VERIFIED: `promise-client.js`]. On unmount, abort via `AbortController` in `useEffect` cleanup to avoid setState after unmount (Claude’s discretion on strictness).
- **Error mapping:** Use `ConnectError` from `@connectrpc/connect` and `error.code` / `error.message` (Lit precedent in `grpc-status-modal.ts` lines 470–476). Expose a small serializable type for React state (D-09).

### `useToast` / `ToastContext`

- **Bridge target:** `ToastContainer` methods `success`, `error`, `warning`, `info` with signature `(title, message?, duration?)` returning toast id string [VERIFIED: `toast-container.ts` lines 215–228].
- **Lit precedent:** `filter-panel.ts` resolves the element via `document.querySelector('toast-container')` as `ToastContainer | null` — React hook should null-check and optionally no-op or log if missing.
- **Registration:** `main.ts` uses `import './components/app/toast-container';` so the custom element is defined [VERIFIED: `main.ts` line 2]. **`front-end/src/react/main.tsx` must import the same module** (side effect) or the tag stays undefined even if `react.html` contains `<toast-container></toast-container>`.

### `useFilters`

- **RPC:** `client.listFilters(new ListFiltersRequest({}))` using `ImageProcessorService` from `front-end/src/gen/image_processor_service_connect.ts` (method name `listFilters`, Connect name `ListFilters`).
- **State:** `filters` array from `ListFiltersResponse`, plus `loading`, `error`, `refetch` (D-11).

### `useHealthMonitor`

- **RPC:** `client.checkAcceleratorHealth(new CheckAcceleratorHealthRequest({}))` via `RemoteManagementService` from `front-end/src/gen/remote_management_service_connect.ts`.
- **Healthy semantics:** Match `grpc-status-modal.ts` `getStatusInfo()`: `acceleratorHealth.status === AcceleratorHealthStatus.HEALTHY` [VERIFIED: `grpc-status-modal.ts` lines 589–603]. Note: the modal polls every **5s** when open (`startAutoRefresh`); the hook uses **15–30s** per D-14 — different UX scopes, same RPC.
- **Visibility:** `document.visibilityState`, `visibilitychange` listener; clear interval and optionally abort in-flight request when hidden (D-13).
- **Leaks:** Clear `setInterval` on unmount and on visibility transitions; remove event listener in cleanup.

## Testing

- **Runner:** `cd front-end && npm run test` runs Vitest in watch mode; CI-style: `npx vitest run` [VERIFIED: command executed — 12 files, 179 tests passing].
- **Environment:** `happy-dom` + `src/test-setup.ts` (mocks `logger`, WebRTC globals) [VERIFIED: `vitest.config.ts`].
- **What to mock:**
  - **Prefer mocking at the `PromiseClient` boundary:** wrap tests with a provider that supplies stub clients `{ listFilters: vi.fn(), checkAcceleratorHealth: vi.fn() }` so hooks do not hit the network.
  - **Alternative:** mock `@connectrpc/connect-web` `createConnectTransport` only in provider tests that assert “single transport” wiring — use sparingly to avoid coupling tests to transport internals.
- **Custom elements in tests:** Follow `filter-panel.test.ts`: `import '../components/app/toast-container'` and append `<toast-container>` to `document.body` when testing `useToast` integration.
- **React Testing Library:** Not required by Phase 2 research scope; optional for hook wrappers if the planner wants DOM assertions (REQUIREMENTS TEST-* are v1.1+).

## Pitfalls

| Pitfall | What goes wrong | Mitigation |
|---------|-----------------|------------|
| **Double transport** | Duplicate connections, inconsistent interceptors, harder debugging | One factory; clients share transport (D-01, D-03). |
| **Toast element missing or undefined** | Silent failures or runtime errors when calling methods on `null` | `react.html` host (D-06) **plus** side-effect import of `toast-container.ts` in React entry. |
| **Health polling leaks** | Background timers and state updates after unmount | `useEffect` cleanup: `clearInterval`, `removeEventListener`, abort `AbortController` if used. |
| **Strict Mode double effects** | React 18+ StrictMode runs effects twice in dev | Idempotent fetches or guard with ref; avoid duplicate intervals without cleanup. |
| **Telemetry parity** | Traces missing on `/react` if `telemetryService.initialize()` never runs | Accept empty trace headers initially, or add optional bootstrap in `main.tsx` mirroring Lit init (open design choice). |
| **Confusing health with processor status** | `getProcessorStatus` vs `checkAcceleratorHealth` | HOOK-05 is **only** `CheckAcceleratorHealth` per D-12 (modal still uses both for different UI concerns). |

## Validation Architecture

> Nyquist enabled: `workflow.nyquist_validation: true` [VERIFIED: `.planning/config.json`]

### Test framework

| Property | Value |
|----------|-------|
| Framework | Vitest ^1.2.0 [VERIFIED: `front-end/package.json`] |
| Config | `front-end/vitest.config.ts` |
| Environment | `happy-dom` |
| Setup | `front-end/src/test-setup.ts` |
| Quick run (watch) | `cd front-end && npm run test` |
| Full run (CI) | `cd front-end && npx vitest run` |
| Coverage | `cd front-end && npm run test:coverage` |

### Phase requirements → tests

| Req ID | Behavior to prove | Test type | Automated command | File status |
|--------|-------------------|-----------|-------------------|-------------|
| HOOK-01 | Consumers receive clients from context only | Unit (provider + hook throws without provider) | `npx vitest run src/react/**/*.test.tsx` | Add in Wave 0 |
| HOOK-02 | `useAsyncGRPC` transitions loading → data/error; `refetch` re-runs call | Unit | `npx vitest run src/react/**/*.test.tsx` | Add |
| HOOK-03 | `useToast` / `ToastContext` invokes `.success`/`.error` on mounted `toast-container` | Unit (integration-style with happy-dom) | `npx vitest run src/react/**/*.test.tsx` | Add |
| HOOK-04 | `useFilters` calls `listFilters` and exposes list + refetch | Unit (mock client) | `npx vitest run src/react/**/*.test.tsx` | Add |
| HOOK-05 | `useHealthMonitor` polls on interval, pauses when `document.visibilityState === 'hidden'`, maps `AcceleratorHealthStatus` | Unit (fake timers + visibility mock) | `npx vitest run src/react/**/*.test.tsx` | Add |

### Sampling rate

- **Per task / wave:** `npx vitest run` scoped to new React test files.
- **Phase gate:** `cd front-end && npx vitest run` full green before `/gsd-verify-work`.

### Manual checks (recommended)

- Load `/react` with dev stack: confirm toasts render visually and Connect calls succeed against Go (same origin).
- Throttle or stop backend: confirm `useHealthMonitor` reflects down within one poll interval (15–30s).

### Wave 0 gaps

- [ ] Create `src/react/**/*.test.tsx` for HOOK-01–HOOK-05 (currently only `webrtc-stub-import.test.tsx` under `src/react/`).
- [ ] Add test utilities: `renderWithGrpcProviders(ui, { mocks })` helper to avoid duplication.

## Security domain (lightweight)

| ASVS area | Applies | Notes |
|-----------|---------|-------|
| V5 Input validation | Partial | Protobuf messages for unary calls; no user-controlled raw strings beyond what Phase 3 sends. |
| V9 Communication | Yes | Same-origin `baseUrl: window.location.origin` [VERIFIED: `config-service.ts`]; cookies/`/grpc` behavior unchanged from Lit. |

No new auth/secrets in this phase [ASSUMED: matches existing Lit threat model].

## Sources

### Primary (HIGH)

- Codebase: `config-service.ts`, `toast-container.ts`, `grpc-status-modal.ts`, `remote-management-service.ts`, `image_processor_service_connect.ts`, `remote_management_service_connect.ts`, `react.html`, `main.tsx`, `vitest.config.ts`, `package.json`
- Installed Connect ES 1.7: `front-end/node_modules/@connectrpc/connect/dist/esm/promise-client.js` (unary `signal` / `timeoutMs`)
- Context7 library `/connectrpc/connect-es` — `ConnectError`, client call options, cancellation patterns [CITED: https://context7.com/connectrpc/connect-es/llms.txt]

### Secondary (MEDIUM)

- npm registry latest major for drift awareness: `npm view @connectrpc/connect version` → 2.1.1 (contrast with pinned 1.7.0)

## Assumptions log

| # | Claim | Risk if wrong |
|---|--------|----------------|
| A1 | No new backend RPCs or paths for Phase 2 | Hook implementation breaks if protos change out of band |
| A2 | ASVS scope above is sufficient for “infrastructure only” phase | Security review may ask for more explicit RPC error handling rules in Phase 3 |

## Open Questions (RESOLVED)

1. **Telemetry bootstrap on `/react` — RESOLVED:** **Defer** calling `telemetryService.initialize()` (and Lit-style `streamConfigService.initialize()`) in Phase 2. Interceptors still inject headers per D-02; empty trace headers when telemetry is off are acceptable until a later milestone aligns `/react` bootstrap with `front-end/src/main.ts`.

2. **Context surface area — RESOLVED:** **Single** React context (or one provider object) exposing both `ImageProcessorService` and `RemoteManagementService` clients, per D-03 and `02-01-PLAN.md`; no split contexts in Phase 2.

## Environment availability

Step 2.6 (external CLIs): **skipped for automated unit tests** — happy-dom does not require Go or CUDA. **Manual / integration:** backend reachable on configured dev origin (`VITE_API_ORIGIN` / same-origin policy per SCAF-01).

| Dependency | Required by | Available | Fallback |
|------------|-------------|-----------|----------|
| Node + npm | Vitest | ✓ (v22.22.2 probed in session) | — |
| Go + TLS dev server | Manual Connect verification | environment-specific | document in plan |
| GPU / Jetson | Health RPC semantics | optional | mock `CheckAcceleratorHealth` in tests |

## RESEARCH COMPLETE
