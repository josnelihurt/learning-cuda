# Phase 1: Scaffold and Infrastructure - Research

**Researched:** 2026-04-12  
**Domain:** Vite 5 MPA, React + Lit coexistence, Nginx path routing, Vitest globals  
**Confidence:** HIGH (repo layout + Vite/Vitest docs verified); MEDIUM (dev “single host” vs split ports; `/lit` routing gap)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Frontend File Layout
- **D-01:** Frontend lives in separate `front-end/` directory (not in `webserver/`)
- **D-02:** Frontend has its own `package.json`, `vite.config.ts`, build scripts, deployment configs
- **D-03:** React code structure: `front-end/src/react/` nested inside existing `src/` directory
- **D-04:** React imports shared code via relative paths from `../gen/` and `../infrastructure/`

#### Vite MPA Configuration
- **D-05:** Single `vite.config.ts` with `rollupOptions.input` pointing to two HTML files (`templates/index.html` for Lit, `templates/react.html` for React)
- **D-06:** Vite manifest uses HTML paths as keys (`index.html`, `react.html`) — these match Nginx static file structure
- **D-07:** Split tsconfigs: `tsconfig.lit.json` (with `experimentalDecorators: true`) and `tsconfig.react.json` (clean, no decorators)
- **D-08:** Both Lit and React bundles output to same `static/js/dist/` directory under Nginx root

#### Production Serving Architecture
- **D-09:** Nginx serves static files directly from `/usr/share/nginx/html` — NO Go template parsing in production
- **D-10:** Traefik routes HTTP/HTTPS to Nginx, which proxies `/api/*` and `/ws` to Go backend
- **D-11:** Go backend ONLY handles API requests, gRPC calls, and WebSocket connections — NO frontend routing
- **D-12:** Path-based routing via Nginx rewrite rules: `/` → `/index.html`, `/react.html` → `/react.html`

#### Development Architecture
- **D-13:** Go dev server proxies all non-API requests to Vite dev server at `localhost:3000`
- **D-14:** Vite dev server serves both HTML entry points automatically via HMR
- **D-15:** Go dev handler is route-aware: returns correct Vite entry script tags for `/lit` and `/react`

#### React App Shell
- **D-16:** Minimal shell with navbar matching Lit's branding ("CUDA Image Processor")
- **D-17:** Reuses existing `static/css/main.css` for consistent look-and-feel
- **D-18:** No feature logic, hooks, or component wiring — just shell structure ready for Phase 2

#### WebRTC Test Stubs
- **D-19:** Minimal global stubs in `front-end/src/test-setup.ts`: `global.RTCPeerConnection = vi.fn()` and `navigator.mediaDevices = { getUserMedia: vi.fn() }`
- **D-20:** Stubs are simple `vi.fn()` mocks — enough to prevent import crashes

#### Build Validation
- **D-21:** All 4 success criteria are hard blockers: (1) `/react` loads React shell, (2) `/lit` loads Lit unchanged, (3) `npm run dev` and `npm run build` succeed with dual entries, (4) WebRTC stubs allow React test imports

#### Frontend Separation Decisions
- **D-22:** Frontend is now a separate service — owns its own build, deployment, and runtime
- **D-23:** Traefik routes frontend requests to Nginx container
- **D-24:** Go backend no longer serves frontend files — only handles API/gRPC/WebSocket
- **D-25:** Lit code remains in `front-end/src/` alongside React during migration — will be removed after React parity

### Claude's Discretion

- Exact Nginx rewrite rules for root redirect and `/react.html` handling
- Vite HMR proxy configuration for both entry points
- Production deployment strategy for static files with Nginx
- Error states for `/react` route (e.g., build missing)
- How to structure `tsconfig.base.json` shared settings

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SCAF-01 | Developer can run React frontend at `/react` and Lit frontend at `/lit` from the same Go server simultaneously | **Current repo:** Go (`webserver/pkg/app/app.go`) registers API, `/ws`, `/data/`, Flipt — **no** HTML/static frontend routes [VERIFIED: codebase]. Production path is Traefik → `web-frontend` (Nginx) → static `dist`; `front-end/nginx.conf` has `location = /react` → `react.html`, but **no** `/lit` alias yet (Lit is served via `/` → `try_files` → `index.html`) [VERIFIED: codebase]. CONTEXT D-13/D-15 describe a Go→Vite dev proxy **not present** in current Go code — planner must either implement it or redefine “same server” as “same logical app URL” (e.g. Nginx or Traefik in dev) with explicit success criteria. |
| SCAF-02 | Vite is configured as a multi-page app (MPA) with separate entry points for Lit and React builds | **Implemented:** `front-end/vite.config.ts` sets `build.rollupOptions.input` to `index.html` and `react.html`, `build.manifest: true`, single `outDir: 'dist'` [VERIFIED: codebase]. Vite documents MPA via multiple HTML files in `rollupOptions.input` [CITED: https://github.com/vitejs/vite/blob/v5.4.21/docs/guide/build.md]. Vite **ignores** the object keys for HTML entries and uses resolved file paths for output names [CITED: same doc]. |
</phase_requirements>

## Summary

The repository already matches the **separate `front-end/`** layout: dual HTML entries (`front-end/index.html`, `front-end/react.html`), `vite.config.ts` MPA input, `@vitejs/plugin-react`, split TypeScript configs (`tsconfig.json` for Lit decorators vs `tsconfig.react.json` for React), React shell under `src/react/`, and Vitest `setupFiles` pointing at `src/test-setup.ts` with WebRTC globals stubbed.

**Gaps vs written plans and CONTEXT:** (1) CONTEXT still references HTML under `templates/` and output under `static/js/dist/` — the **live** project uses repo-root `index.html` / `react.html` and `build.outDir: 'dist'` (copied to Nginx root by `front-end/Dockerfile`) [VERIFIED: codebase]. (2) **SCAF-01** requires `/lit` and “same Go server”; **current** dev workflow is **split**: `./scripts/dev/start.sh` → Go on `:8443`, `./scripts/dev/start-frontend.sh` → Vite on `:3000` [VERIFIED: `scripts/dev/start.sh`, `start-frontend.sh`]. Go does not proxy the UI. (3) Nginx serves `/react` via rewrite to `react.html` but has **no** `location = /lit` — today Lit is the default app at `/` [VERIFIED: `front-end/nginx.conf`]. (4) Roadmap success criterion #3 mentions “separate output directories”; the **implemented** Vite pattern is **one** `dist/` with two HTML entry outputs — interpret as “separate entry artifacts,” not two `outDir`s, unless the user changes D-08.

**Primary recommendation:** Treat **Nginx (production)** and **Vite dev (development)** as the owners of `/react` and `/lit` path semantics; add explicit **dev** rewrites (Vite plugin or `server` middleware) so `/react` and `/lit` work without `.html` suffix, mirroring Nginx. Reconcile “same Go server” with either a **small Go reverse proxy** for dev only or an **updated requirement** that the unified entry is Traefik/Nginx in compose — document the choice in PLAN.md.

## Project Constraints (from .cursor/rules/)

No files were present under `.cursor/rules/` in this workspace at research time — no additional machine-enforceable directives beyond repository norms (see `CLAUDE.md` for build/test commands).

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vite | `^5.0.10` (lockfile resolves 5.x); **latest registry** `8.0.8` [VERIFIED: `npm view vite version`] | Bundler + dev server for Lit + React MPA | Project is pinned to Vite 5; do not jump major without intentional upgrade |
| @vitejs/plugin-react | `^4.7.0` in repo; **latest registry** `6.0.1` [VERIFIED: `npm view @vitejs/plugin-react version`] | React Fast Refresh, JSX transform | Roadmap/STATE: stay on **v4** with Vite 5 (v6 targets newer Vite — [ASSUMED] per STATE.md; confirm before upgrade) |
| React / React DOM | `^19.2.5` [VERIFIED: `package.json`, `npm view react version`] | React shell | Already adopted in `front-end/` |
| Vitest | `^1.2.0`; **latest registry** `4.1.4` [VERIFIED: `npm view vitest version`] | Unit tests | Matches existing `vitest.config.ts` / happy-dom setup |
| happy-dom | `^12.10.3` [VERIFIED: `package.json`] | DOM env for Vitest | Already configured in `vitest.config.ts` |

### Supporting

| Library | Purpose | When to Use |
|---------|---------|-------------|
| Lit `^3.1.0` | Existing components | Until React parity (Phase 5+) |
| ConnectRPC clients in `src/gen/` | Shared API surface | React imports per D-04 |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Single `dist/` MPA | Two Vite projects / two builds | Simpler deploy today; dual project adds coordination overhead |
| Go dev proxy | Dev-only Traefik+Nginx | Heavier local setup; closer to prod |

**Installation:** Already satisfied in `front-end/package.json`; use `npm ci` in Docker, `npm install` locally.

## Architecture Patterns

### Current repository layout (authoritative)

```text
front-end/
├── index.html              # Lit entry (Vite root)
├── react.html              # React entry
├── vite.config.ts          # MPA + proxy to Go API
├── vitest.config.ts
├── nginx.conf              # Production static + /api, /ws, /react rewrite
├── Dockerfile              # buf generate + npm run build → Nginx image
├── src/
│   ├── main.ts             # Lit bootstrap
│   ├── react/main.tsx      # React bootstrap
│   ├── react/App.tsx       # Shell
│   ├── components/         # Lit
│   ├── gen/, infrastructure/  # Shared
│   └── test-setup.ts
└── dist/                   # Build output (gitignored)
```

### Pattern 1: Vite MPA (two HTML entries)

**What:** Multiple top-level HTML files as Rollup inputs; dev server serves each; build emits `dist/index.html`, `dist/react.html`, shared chunks, and `.vite/manifest.json` when `build.manifest: true`.

**When to use:** Exactly this phase — Lit and React side by side.

**Example:**

```javascript
// Source: https://github.com/vitejs/vite/blob/v5.4.21/docs/guide/build.md
import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        nested: resolve(__dirname, 'nested/index.html'),
      },
    },
  },
})
```

**Project mapping:** `front-end/vite.config.ts` already uses `main` + `react` keys; output filenames follow **resolved HTML paths** (`index.html`, `react.html`), not the keys [CITED: Vite build.md above].

### Pattern 2: Vitest global stubs via `setupFiles`

**What:** `setupFiles` run in the test worker before each test file — correct place for `RTCPeerConnection` / `navigator.mediaDevices` shims.

**When to use:** Any module that touches WebRTC APIs during import in tests.

**Example:**

```typescript
// Source: https://github.com/vitest-dev/vitest/blob/main/docs/config/setupfiles.md
// Paths resolved relative to Vitest root; runs before each test file in the same process.
```

**Project mapping:** `front-end/vitest.config.ts` → `setupFiles: ['src/test-setup.ts']` [VERIFIED: codebase].

### Anti-Patterns to Avoid

- **`@vitejs/plugin-react` default `include`:** README states default includes `.js`, `.jsx`, `.ts`, `.tsx` [CITED: https://raw.githubusercontent.com/vitejs/vite-plugin-react/main/packages/plugin-react/README.md]. That can run React Fast Refresh pipeline on **all** `.ts` files including Lit — if decorators or non-React `.ts` break, narrow with `react({ include: /\.(tsx|jsx)$/ })` or explicit globs [MEDIUM confidence — verify if Lit HMR/regressions appear].
- **Assuming Go serves the UI in dev:** Current code does not; scripts tell developers to run Vite separately [VERIFIED: `scripts/dev/start.sh`].
- **Relying on `/react` without rewrite on Vite dev:** Nginx rewrites `/react` → `react.html` [VERIFIED: `nginx.conf`]; Vite dev typically serves `/react.html` directly — `/react` may 404 unless a small dev middleware/plugin adds the same rewrite [ASSUMED behavior; verify with manual request].

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multi-entry bundling | Separate ad-hoc Rollup configs | Vite `rollupOptions.input` + HTML entries | Vite handles HMR, CSS, manifest [CITED: Vite build.md] |
| WebRTC in tests | Real peer connections in unit tests | `vi.fn()` globals in `setupFiles` | No headless WebRTC; matches D-19/D-20 |
| Path routing in production | Go `http.FileServer` for SPA | Nginx `try_files` + exact rewrites | Aligns with D-09–D-12 and current `nginx.conf` |

**Key insight:** Edge routing is **already** split: static UI at Nginx, API at Go — plans should not reintroduce Go template HTML for production without an explicit decision change.

## Common Pitfalls

### Pitfall 1: Doc drift (`templates/` vs repo root HTML)

**What goes wrong:** Plans reference `templates/index.html` while files live at `front-end/index.html`.

**Why it happens:** Frontend was moved out of `webserver/web/` into `front-end/` without updating all planning artifacts.

**How to avoid:** Anchor tasks to paths from `01-CONTEXT.md` canonical refs **after** verifying on disk.

**Warning signs:** Plan steps that `cd` to missing directories.

### Pitfall 2: `/lit` and `/react` parity between dev and prod

**What goes wrong:** Production serves correct HTML; Vite dev only serves `/react.html` and `/index.html`.

**Why it happens:** Pretty paths need explicit rewrites (Nginx has `/react`; dev may lack `/lit` and `/react`).

**How to avoid:** One table mapping **path → HTML file** for Nginx, Vite middleware, and any future Go proxy.

**Warning signs:** E2E or manual tests pass on `localhost:3000/react.html` but fail on `/react`.

### Pitfall 3: “Same Go server” vs actual dev topology

**What goes wrong:** SCAF-01 interpreted as Go serving both routes while implementation uses Nginx + separate Vite port.

**Why it happens:** CONTEXT mixes production (Nginx) with a Go dev proxy (D-13) that is not implemented in `webserver/`.

**How to avoid:** Define the **single URL** developers use (e.g. `https://localhost:8443` vs `https://localhost:3000`) and implement the missing proxy or update the requirement.

**Warning signs:** No `vite` or `reverse` references under `webserver/pkg/`.

## Code Examples

### MPA `rollupOptions.input` (official)

See **Pattern 1** above [CITED: Vite v5.4.21 build guide].

### Vitest `setupFiles` (official)

Vitest: `setupFiles` paths resolve relative to project root and run before each test file [CITED: Vitest setupfiles.md via Context7].

### Nginx pretty path for React (this repo)

```nginx
# Source: front-end/nginx.conf (verbatim pattern)
location = /react {
    rewrite ^ /react.html break;
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Lit UI under `webserver/web/` | Standalone `front-end/` service + Docker Nginx | Migration (per CONTEXT) | Go no longer serves frontend static assets |
| Single HTML entry | Vite MPA `index.html` + `react.html` | Phase 1 scaffold | Shared chunks + two shells |

**Deprecated/outdated:** Planning references to `webserver/pkg/interfaces/statichttp/development_handler.go` — file **not present** in tree; do not plan against it without restoring or replacing [VERIFIED: glob + read `statichttp/handler.go`].

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Vite dev returns 404 for `/react` unless middleware added | Pitfalls | Manual verification fails for success criterion #1 on `:3000` |
| A2 | Staying on `@vitejs/plugin-react` v4 with Vite 5 is required (avoid v6 until Vite upgrade) | Standard Stack | Accidental upgrade breaks build — confirm with maintainer |
| A3 | “Separate output directories” in roadmap means separate entry outputs under one `dist/`, not two `outDir`s | Summary | Wrong task scope if user wanted literal two folders |

**If this table is empty:** N/A — assumptions listed above need confirmation where marked.

## Open Questions

1. **What is the canonical “single server” URL for SCAF-01 in local dev?**
   - What we know: Go `:8443` and Vite `:3000` are separate today [VERIFIED: scripts].
   - What’s unclear: Whether to implement Go→Vite proxy (D-13) or standardize on Traefik/Nginx for local full stack.
   - Recommendation: Pick one URL for acceptance tests and document in PLAN.md.

2. **Should `/` redirect to `/react` (per CONTEXT “Specific Ideas”) while Lit lives at `/lit`?**
   - What we know: Nginx currently uses `try_files` → `index.html` for `/` [VERIFIED: `nginx.conf`].
   - What’s unclear: Whether root redirect would break existing bookmarks or E2E that assume Lit at `/`.
   - Recommendation: Align with REQUIREMENTS success criteria and E2E base URLs before changing rewrites.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Node.js | `npm run dev` / `vite build` | ✓ | v22.22.2 (agent env) | Use `node:20` in Docker as in `front-end/Dockerfile` [VERIFIED: Dockerfile] |
| npm | install / CI | ✓ | 10.9.7 | — |
| Go toolchain | Backend API for Vite proxy | ✓ | go1.25.7 | — |
| Docker/Podman | Compose, image build | ✓ | podman 4.9.3 | — |
| Nginx (host) | Local prod-like serve | ✗ (not in PATH) | — | Use `web-frontend` container or `vite preview` |
| `.secrets/localhost+2*.pem` | Vite HTTPS dev server | — | — | Script checks presence [VERIFIED: `start-frontend.sh`] |

**Missing dependencies with no fallback:**

- None for writing code; **host Nginx** is optional if using containerized frontend.

**Missing dependencies with fallback:**

- Local Nginx → use Docker image built from `front-end/Dockerfile`.

## Validation Architecture

> Nyquist validation is enabled (`workflow.nyquist_validation: true` in `.planning/config.json`) [VERIFIED: `.planning/config.json`].

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Vitest `^1.2.0` + happy-dom `^12.10.3` [VERIFIED: `package.json`, `vitest.config.ts`] |
| Config file | `front-end/vitest.config.ts` |
| Quick run command | `cd front-end && npx vitest run --passWithNoTests` (or targeted path) |
| Full suite command | `cd front-end && npm run test` (watch) / `npx vitest run` (CI-style) |
| E2E (optional gate) | `cd front-end && npm run test:e2e` (Playwright) [VERIFIED: `package.json`] |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| SCAF-01 | `/react` and `/lit` resolve to correct shells on the **agreed** dev/prod host | integration / e2e / manual | Playwright navigates URLs OR curl + assert HTML shell markers | Add spec under `front-end/tests/e2e/` if not present — **Wave 0** |
| SCAF-02 | `vite build` emits both HTML entries and manifest | build + smoke | `cd front-end && npm run build && test -f dist/index.html && test -f dist/react.html` | ✅ Wave 0 command |
| WebRTC stubs | Importing modules touching WebRTC does not throw in Vitest | unit | `cd front-end && npx vitest run src/...` for a React test importing shared infra | Extend `src/react/*.test.tsx` if needed — **Wave 0** |

### Sampling Rate

- **Per task commit:** `cd front-end && npx vitest run` (targeted if possible) + `npm run build` when touching Vite/React.
- **Per wave merge:** `cd front-end && npx vitest run` + `npm run build`.
- **Phase gate:** `scripts/test/unit-tests.sh` (if includes frontend) or repo CI equivalent green before `/gsd-verify-work`.

### Wave 0 Gaps

- [ ] E2E or scripted HTTP checks for `/react` and `/lit` on the **same origin** as required by finalized SCAF-01 interpretation.
- [ ] Minimal React unit test proving `test-setup.ts` stubs allow importing a module that references `RTCPeerConnection` (if not already covered).

*(Build verification for dual HTML: `npm run build` + assert `dist/react.html` and `dist/index.html` exist.)*

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | N/A this phase |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | partial | No new user input — static routes; keep API proxy paths unchanged |
| V6 Cryptography | partial | TLS: Vite dev uses certs from `.secrets/`; production TLS via Traefik [VERIFIED: `vite.config.ts`, `docker-compose.yml` labels] |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Open dev proxy to arbitrary targets | Spoofing | Pin `VITE_API_ORIGIN` / proxy `target` to known backend hosts [VERIFIED: `vite.config.ts` uses env] |
| Missing `secure` / TLS verify in dev proxy | Tampering | Acceptable for local trusted certs; do not copy `secure: false` patterns to production edge config without review |

## Sources

### Primary (HIGH confidence)

- [Context7 `/vitejs/vite/v5.4.21`] — MPA `rollupOptions.input`, HTML output naming, manifest
- [Context7 `/vitest-dev/vitest`] — `setupFiles` lifecycle vs `globalSetup`
- [CITED: https://raw.githubusercontent.com/vitejs/vite-plugin-react/main/packages/plugin-react/README.md] — default `include` file types for plugin-react
- [VERIFIED: codebase] — `front-end/vite.config.ts`, `package.json`, `nginx.conf`, `webserver/pkg/app/app.go`, `webserver/pkg/interfaces/statichttp/handler.go`, `scripts/dev/start.sh`, `start-frontend.sh`

### Secondary (MEDIUM confidence)

- [VERIFIED: `npm view`] — registry latest versions for vite, plugin-react, vitest (contrast with repo pins)

### Tertiary (LOW confidence)

- [ASSUMED] — Vite dev server behavior for bare `/react` path without custom middleware

## Metadata

**Confidence breakdown:**

- Standard stack: **HIGH** — matches `package.json` + registry spot-check
- Architecture: **MEDIUM** — codebase clear; dev/prod routing story incomplete vs CONTEXT D-13
- Pitfalls: **HIGH** — doc path drift and `/lit` gap are observable in repo

**Research date:** 2026-04-12  
**Valid until:** ~2026-05-12 (stable stack); sooner if Vite major upgrade is attempted
