# Phase 1: Scaffold and Infrastructure - Context

**Gathered:** 2026-04-12 (updated 2026-04-12)
**Status:** Ready for planning

<domain>
## Phase Boundary

Developer can load the React frontend at `/react` and the Lit frontend at `/lit` simultaneously from a single Vite build and Go server. WebRTC APIs are stubbed in test-setup so React tests can import WebRTC-using modules without crashing. No feature logic — this is pure infrastructure.

</domain>

<decisions>
## Implementation Decisions

### React File Layout
- **D-01:** React code lives in `src/react/` nested inside the existing `src/` directory (not a separate top-level directory)
- **D-02:** React imports shared code directly from `../gen/` and `../infrastructure/` using the existing `@` path alias (no re-export layer)
- **D-03:** React tests are co-located with source files (e.g., `src/react/components/Button.test.tsx` next to `Button.tsx`)

### Vite MPA Configuration
- **D-04:** Single `vite.config.ts` with `rollupOptions.input` pointing to two HTML files (`templates/index.html` for Lit, `templates/react.html` for React). Current config uses `src/main.ts` as TS entry — switching to HTML-based MPA changes manifest keys from `"src/main.ts"` to HTML paths
- **D-05:** Split tsconfigs: `tsconfig.lit.json` (with `experimentalDecorators: true`, `useDefineForClassFields: false`) and `tsconfig.react.json` (clean, no decorator options). Shared `tsconfig.base.json` for common settings
- **D-06:** Both Lit and React bundles output to the same `static/js/dist/` directory. `asset_manifest.go` must read multiple manifest keys (currently keys on `"src/main.ts"`)
- **D-07:** `@vitejs/plugin-react@^4` is added to the single Vite config alongside existing Lit support (no plugin conflict expected — JSX transform is handled by the plugin, decorators by tsconfig)

### Go Routing Strategy
- **D-08:** Path-based routing: `/lit/` serves the Lit template, `/react/` serves the React template. Root `/` redirects to `/react` (React is the default experience)
- **D-09:** Separate `templates/react.html` template file (not a shared template with conditionals). React template has `<div id="root">` and React-specific script injection
- **D-10:** In dev mode, Go proxies all non-API requests to Vite dev server — no proxy config changes needed. Vite's MPA mode serves both HTML entry points automatically
- **D-11:** Lit template and route remain completely unchanged — zero regression risk

### React App Shell
- **D-12:** Minimal shell with a navbar matching Lit's branding ("CUDA Image Processor" header) and a placeholder body
- **D-13:** Reuses existing `static/css/main.css` for consistent look-and-feel between `/lit` and `/react`
- **D-14:** No feature logic, hooks, or component wiring — just the shell structure ready for Phase 2 hooks and Phase 3 features

### WebRTC Test Stubs
- **D-15:** Minimal global stubs added to the existing `src/test-setup.ts`: `global.RTCPeerConnection = vi.fn()` and `navigator.mediaDevices = { getUserMedia: vi.fn() }`
- **D-16:** Stubs are simple `vi.fn()` mocks — enough to prevent import crashes. Full typed mocks deferred to Phase 4

### Build Validation
- **D-17:** All 4 success criteria are hard blockers: (1) `/react` loads React shell, (2) `/lit` loads Lit unchanged, (3) `npm run dev` + `npm run build` both succeed with dual entries, (4) WebRTC stubs allow React test imports

### the agent's Discretion
- Exact `asset_manifest.go` refactoring to support multiple manifest keys — confirmed blocker: current code hard-codes `m["src/main.ts"]`; HTML-based MPA changes keys to HTML paths
- Multi-entry dev handler: `DevelopmentAssetHandler.GetScriptTags()` currently returns hard-coded `["/@vite/client", "/src/main.ts"]` — needs route-aware script tag selection for Lit vs React entries
- Template parsing strategy: `NewStaticHandler` currently parses one template at construction; dual routing needs two pre-parsed templates or on-demand selection
- React HTML template boilerplate (meta tags, font imports, etc.)
- Error states for `/react` route (e.g., build missing)
- How to structure `tsconfig.base.json` shared settings

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Vite and Build Configuration
- `webserver/web/vite.config.ts` — Current single-entry Vite config (`input: resolve(__dirname, 'src/main.ts')`), build output settings, dev proxy config
- `webserver/web/tsconfig.json` — Current TypeScript config with Lit decorator settings (`experimentalDecorators: true`, `useDefineForClassFields: false`)
- `webserver/web/package.json` — Frontend dependencies (no React yet), build scripts

### Go Static Serving
- `webserver/pkg/interfaces/statichttp/asset_manifest.go` — Manifest parsing, entry file resolution (hard-codes `m["src/main.ts"]` key — must change for HTML-based MPA)
- `webserver/pkg/interfaces/statichttp/handler.go` — Static handler, route registration, ServeIndex, template execution
- `webserver/pkg/interfaces/statichttp/production_handler.go` — Production asset handler, script tag generation from manifest
- `webserver/pkg/interfaces/statichttp/development_handler.go` — Dev proxy handler, hard-coded script tags (`/@vite/client`, `/src/main.ts`)
- `webserver/pkg/interfaces/statichttp/asset_handler.go` — AssetHandler interface definition
- `webserver/pkg/app/app.go` — Go app bootstrap, route setup, catch-all handler (lines 263-284)

### Frontend Entry Points
- `webserver/web/templates/index.html` — Current Lit HTML template (reference for React template)
- `webserver/web/src/main.ts` — Current Lit entry point (reference only, not modified)
- `webserver/web/src/test-setup.ts` — Current test setup with logger mock (WebRTC stubs added here)

### Project-Level Constraints
- `.planning/ROADMAP.md` §Phase 1 — Phase goal, success criteria, requirements SCAF-01/SCAF-02
- `.planning/REQUIREMENTS.md` §SCAF-01, SCAF-02 — Dual route and MPA requirements
- `.planning/STATE.md` §Blockers/Concerns — Known concerns about asset_manifest.go and plugin-react compatibility

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `webserver/web/src/gen/` — Generated protobuf clients (ConnectRPC). React will import these directly via `@/gen/`
- `webserver/web/src/infrastructure/observability/otel-logger.ts` — OTEL logger. React will import via `@/infrastructure/`
- `webserver/web/static/css/main.css` — Full CSS stylesheet. React template links to this for consistent styling
- `webserver/web/src/application/di/Container.ts` — Lit DI container. React does NOT use this (Context providers instead, per roadmap decision)

### Established Patterns
- **Vite build**: Single entry `src/main.ts` → output `static/js/dist/app.[hash].js`. MPA needs HTML-based input which changes manifest key structure
- **Go template serving**: `StaticHandler` parses `templates/index.html`, injects `ScriptTags` from asset handler, executes template. Single template at construction time
- **Asset manifest**: `loadAssetManifest()` reads `.vite/manifest.json`, looks up `src/main.ts` key. Must handle HTML-path keys after MPA change
- **Dev mode**: `DevelopmentAssetHandler` proxies to Vite dev server at `localhost:3000`, returns hard-coded HMR script tags. Needs route-aware tags for dual entry
- **Production mode**: `ProductionAssetHandler` reads manifest, returns hashed bundle script tags from manifest entry

### Integration Points
- **`vite.config.ts`** — Change `rollupOptions.input` from `src/main.ts` to `[templates/index.html, templates/react.html]`, add `@vitejs/plugin-react` plugin
- **`asset_manifest.go`** — `GetEntryFile()` must support multiple entry points with HTML-path keys instead of `"src/main.ts"`
- **`handler.go`** — `RegisterRoutes()` needs `/lit/` and `/react/` routes, `ServeIndex()` needs template selection based on route
- **`app.go`** — Catch-all handler (line 263) currently serves Lit index for all non-API routes. Must route `/lit/` → Lit, `/react/` → React, `/` → redirect to `/react`
- **`test-setup.ts`** — Add WebRTC global stubs after existing logger mock

</code_context>

<specifics>
## Specific Ideas

- React is the default experience — root `/` redirects to `/react`, not `/lit`
- The split tsconfig approach lets React use modern defaults without Lit's decorator legacy
- `asset_manifest.go` is the key risk — verified that it hard-codes `m["src/main.ts"]`; HTML-based MPA changes this key
- Current `DevelopmentAssetHandler.GetScriptTags()` is hard-coded for Lit entry — needs refactoring for route-aware dual entry

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-scaffold-and-infrastructure*
*Context gathered: 2026-04-12*
