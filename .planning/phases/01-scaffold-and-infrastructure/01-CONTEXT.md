# Phase 1: Scaffold and Infrastructure - Context

**Gathered:** 2026-04-12 (updated 2026-04-13)
**Status:** Ready for planning

<domain>
## Phase Boundary

Developer can load React frontend at `/react` and Lit frontend at `/lit` simultaneously from a single Vite build. Frontend is now a separate service (`front-end/`) decoupled from Go backend. WebRTC APIs are stubbed in test-setup so React tests can import WebRTC-using modules without crashing. No feature logic — this is pure infrastructure.

**Production Architecture:**
- Traefik (edge) → Nginx (static file server) → Go (API/gRPC/WebSocket)
- Frontend files are served directly by Nginx — Go does NOT handle frontend routing in production

**Development Architecture:**
- UI: Vite HTTPS on `https://localhost:3000` via `scripts/dev/start-frontend.sh` (HMR, pretty paths `/react` and `/lit` from `vite.config.ts`)
- API/WebSocket/Connect: Go TLS on `https://localhost:8443` (`./scripts/dev/start.sh`); `VITE_API_ORIGIN` points Vite’s dev proxy at Go
- Go does **not** serve frontend HTML and does **not** reverse-proxy the UI to Vite
</domain>

<decisions>
## Implementation Decisions

### Frontend File Layout
- **D-01:** Frontend lives in separate `front-end/` directory (not in `webserver/`)
- **D-02:** Frontend has its own `package.json`, `vite.config.ts`, build scripts, deployment configs
- **D-03:** React code structure: `front-end/src/react/` nested inside existing `src/` directory
- **D-04:** React imports shared code via relative paths from `../gen/` and `../infrastructure/`

### Vite MPA Configuration
- **D-05:** Single `vite.config.ts` with `rollupOptions.input` pointing to repo-root `front-end/index.html` (Lit) and `front-end/react.html` (React)
- **D-06:** Vite manifest uses HTML paths as keys (`index.html`, `react.html`) — these match Nginx static file structure
- **D-07:** Split tsconfigs: `tsconfig.lit.json` (with `experimentalDecorators: true`) and `tsconfig.react.json` (clean, no decorators)
- **D-08:** Both Lit and React bundles output into the same `front-end/dist/` tree (single `outDir`), with separate HTML entry artifacts and shared chunks — copied to Nginx document root in the `web-frontend` image

### Production Serving Architecture
- **D-09:** Nginx serves static files directly from `/usr/share/nginx/html` — NO Go template parsing in production
- **D-10:** Traefik routes HTTP/HTTPS to Nginx, which proxies `/api/*` and `/ws` to Go backend
- **D-11:** Go backend ONLY handles API requests, gRPC calls, and WebSocket connections — NO frontend routing
- **D-12:** Path-based routing via Nginx rewrite rules: `/` → `/index.html`, `/react.html` → `/react.html`

### Development Architecture
- **D-13:** Split-port dev: run Vite (`scripts/dev/start-frontend.sh`, `https://localhost:3000`) for UI and Go (`scripts/dev/start.sh`, `https://localhost:8443`) for API/WebSocket/Connect. Configure `VITE_API_ORIGIN` so Vite proxies backend paths to Go. **No** Go-served HTML and **no** Go reverse proxy of static/HMR to Vite.
- **D-14:** Vite dev server serves both HTML entries with HMR; `/react` and `/lit` pretty paths are implemented in `vite.config.ts` (dev + preview middleware) to mirror production Nginx.
- **D-15:** Pretty-path routing for `/lit` and `/react` is implemented by **Nginx** in production (`front-end/nginx.conf`) and by **Vite** in local dev/preview — not by Go handlers or Go template injection.

### React App Shell
- **D-16:** Minimal shell with navbar matching Lit's branding ("CUDA Image Processor")
- **D-17:** Reuses existing `static/css/main.css` for consistent look-and-feel
- **D-18:** No feature logic, hooks, or component wiring — just shell structure ready for Phase 2

### WebRTC Test Stubs
- **D-19:** Minimal global stubs in `front-end/src/test-setup.ts`: `global.RTCPeerConnection = vi.fn()` and `navigator.mediaDevices = { getUserMedia: vi.fn() }`
- **D-20:** Stubs are simple `vi.fn()` mocks — enough to prevent import crashes

### Build Validation
- **D-21:** All 4 success criteria are hard blockers: (1) `/react` loads React shell, (2) `/lit` loads Lit unchanged, (3) `npm run dev` and `npm run build` succeed with dual entries, (4) WebRTC stubs allow React test imports

### agent's Discretion
- Exact Nginx rewrite rules for root redirect and `/react.html` handling
- Vite HMR proxy configuration for both entry points
- Production deployment strategy for static files with Nginx
- Error states for `/react` route (e.g., build missing)
- How to structure `tsconfig.base.json` shared settings

### Frontend Separation Decisions
- **D-22:** Frontend is now a separate service — owns its own build, deployment, and runtime
- **D-23:** Traefik routes frontend requests to Nginx container
- **D-24:** Go backend no longer serves frontend files — only handles API/gRPC/WebSocket
- **D-25:** Lit code remains in `front-end/src/` alongside React during migration — will be removed after React parity

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Production Infrastructure
- `front-end/nginx.conf` — Nginx static file server config, proxy rules for `/api/*`, `/ws`, and path rewrites
- `front-end/Dockerfile` — Multi-stage build: proto-tools → builder → Nginx runtime
- `docker-compose.yml` — Service orchestration: Traefik → Nginx (frontend) + Go (backend)
- `traefik-config.yml` — TLS configuration and routing rules for Traefik edge router

### Vite and Build Configuration
- `front-end/vite.config.ts` — MPA config with HTML entry points, build settings, dev proxy config
- `front-end/tsconfig.lit.json` — TypeScript config for Lit with decorator support
- `front-end/tsconfig.react.json` — TypeScript config for React (clean, no decorators)
- `front-end/package.json` — Frontend dependencies, build scripts, Vite config

### Go Backend Integration
- `webserver/pkg/app/app.go` — Go app bootstrap; API/WebSocket/Connect registration (no frontend HTML serving)
- `webserver/pkg/interfaces/statichttp/handler.go` — Static/data routes used by the backend; **not** used to serve the React/Lit MPA HTML
- Note: There is **no** `development_handler.go` in-tree; dev UI is Vite-only on `:3000`

### Frontend Entry Points
- `front-end/index.html` — Lit MPA HTML entry
- `front-end/react.html` — React MPA HTML entry
- `front-end/src/test-setup.ts` — Test setup with WebRTC global stubs

### Project-Level Constraints
- `.planning/ROADMAP.md` §Phase 1 — Phase goal, success criteria, requirements SCAF-01/SCAF-02
- `.planning/REQUIREMENTS.md` §SCAF-01, SCAF-02 — Dual route and MPA requirements
- `.planning/PROJECT.md` — Frontend separation as service, migration goal, React learning objectives

### Build Artifacts
- `front-end/dist/index.html` — Built Lit HTML entry
- `front-end/dist/react.html` — Built React HTML entry
- `front-end/dist/.vite/manifest.json` — Vite manifest with HTML path keys
- `front-end/dist/static/` — CSS, JS bundles, images served by Nginx

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `front-end/src/gen/` — Generated protobuf clients (ConnectRPC). React imports directly via `../gen/`
- `front-end/src/infrastructure/` — Shared infrastructure code (logger, telemetry). React imports via `../infrastructure/`
- `front-end/src/components/` — Existing Lit components — may be ported to React in later phases
- `front-end/static/css/main.css` — Full CSS stylesheet. React template links to this for consistent styling

### Established Patterns
- **Vite MPA**: Single config with HTML-based `rollupOptions.input` → produces `index.html`, `react.html` + manifest with HTML path keys
- **Production serving**: Nginx serves static files directly, Traefik routes edge traffic, Go handles only backend logic
- **Dev serving**: Vite on `:3000` serves UI + proxies API paths to Go on `:8443` per `vite.config.ts` and `VITE_API_ORIGIN`
- **TypeScript configs**: Split configs allow Lit (decorators) and React (clean) to coexist in same frontend

### Integration Points
- **`vite.config.ts`** — Already configured for MPA with HTML entry points
- **`nginx.conf`** — Proxy rules to Go backend for `/api/*`, `/ws`, path rewrites for frontend routing
- **`docker-compose.yml`** — Traefik → Nginx → Go service mesh
- **`test-setup.ts`** — WebRTC stubs added for React test compatibility
- **Go app routing** — Go never serves the MPA HTML; UI routing is Nginx (prod) or Vite (dev)

</code_context>

<specifics>
## Specific Ideas

- Default landing remains Lit at `/` via Nginx `try_files` / Vite root unless a future requirement changes the product default
- Production uses Nginx static serving, not Go templates — Go only handles API/gRPC/WebSocket
- Dev mode uses split ports: Vite for UI and HMR, Go for backend; pretty `/react` and `/lit` match prod via Vite middleware
- The split tsconfig approach lets React use modern TypeScript defaults without Lit's decorator legacy
- Frontend is a separate service (`front-end/`) with its own deployment (Nginx) and build pipeline

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---
*Phase: 01-scaffold-and-infrastructure*
*Context gathered: 2026-04-13 (updated with production architecture)*
