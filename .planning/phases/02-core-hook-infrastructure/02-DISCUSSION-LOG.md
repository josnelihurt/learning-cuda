# Phase 2: Core Hook Infrastructure - Discussion Log

> **Audit trail only.** Decisions live in `02-CONTEXT.md`.

**Date:** 2026-04-13  
**Phase:** 02 — Core Hook Infrastructure  
**Areas discussed:** gRPC transport & clients, Toasts, useAsyncGRPC contract, useHealthMonitor (and useFilters alignment)

---

## Area: gRPC transport and service clients

| Option | Description | Selected |
|--------|-------------|----------|
| Shared transport + Context | One `createConnectTransport`, typed clients in Context, telemetry interceptors like Lit | ✓ |
| Per-service transports | Mirror each Lit service class owning its own transport | |
| Ad-hoc clients in components | Create transport/clients inline | |

**User's choice:** Discuss all areas; lock **shared transport + Context** with Lit-aligned interceptors and `window.location.origin`.

**Notes:** HOOK-01 satisfied without exposing Lit DI.

---

## Area: Toasts (HOOK-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Bridge to `<toast-container>` | `querySelector` + `ToastContainer` methods; parity with Lit | ✓ |
| React-only toast UI | New components; possible drift from Lit until Phase 5 | |

**User's choice:** **Bridge** + ensure **`toast-container` in `react.html` (or shell)** so `/react` has a host element.

---

## Area: useAsyncGRPC contract

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal `{ data, loading, error }` only | Roadmap literal | |
| Extended with `refetch` / `execute` | Supports mutations and manual retries without second hook | ✓ |

**User's choice:** Extend with **`refetch` or imperative `execute`** as in CONTEXT D-08.

---

## Area: useHealthMonitor + filters context

| Topic | Decision |
|-------|----------|
| Health RPC | `RemoteManagementService.checkAcceleratorHealth`; align with `grpc-status-modal` / `AcceleratorHealthStatus` |
| Visibility | Pause polling when tab hidden |
| Interval | Implementation choice; ~15–30s starting point |
| useFilters | `ImageProcessorService` `listFilters` via shared clients |

**User's choice:** As summarized; folded into CONTEXT D-10–D-14.

---

## Claude's Discretion

- Provider file layout, hook internals, tests, optional AbortSignal — per CONTEXT.

## Deferred ideas

- React-native toast only if bridge proves inadequate (later phase / backlog).
- TanStack Query as mandatory layer — not Phase 2.
