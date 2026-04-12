# Architecture Research

**Domain:** React frontend migration — brownfield addition to existing CUDA media processing platform
**Researched:** 2026-04-12
**Confidence:** HIGH (based on direct codebase analysis, not assumptions)

## Standard Architecture

### System Overview

```
Browser
  ├── /lit/* → Lit SPA (existing)
  │     └── index.html (Go template: webserver/web/templates/index.html)
  │           └── src/main.ts  (Lit Web Components + DI container)
  │
  └── /react/* → React SPA (new)
        └── react/index.html (new Go template)
              └── src/react-main.tsx  (React app entry point)

Go HTTP Server (webserver/pkg/app/app.go)
  ├── /static/          → StaticHandler.ServeStatic (CSS, JS dist)
  ├── /data/            → StaticHandler.ServeData
  ├── /ws               → WebSocket handler
  ├── /ws/webrtc-signaling → WebRTC signaling
  ├── /grpc             → gRPC proxy (dev) / direct (prod)
  ├── /cuda_learning.*  → ConnectRPC handlers (Vanguard)
  ├── /com.jrb.*        → ConnectRPC handlers (Vanguard)
  ├── /api/             → REST via Vanguard transcoder
  ├── /health           → Health check
  ├── /lit/*            → ServeIndex (Lit template) [NEW ROUTE]
  ├── /react/*          → ServeIndex (React template) [NEW ROUTE]
  └── /                 → ServeIndex (Lit template — backward compat)

Vite Build (webserver/web/)
  ├── rollupOptions.input.lit   → src/main.ts     → static/js/dist/lit/
  └── rollupOptions.input.react → src/react-main.tsx → static/js/dist/react/
```

### Component Responsibilities

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| `vite.config.ts` | Dual-entry MPA build, dev server proxy | Modified — add second `input` key |
| `src/react-main.tsx` | React app bootstrap, Context setup | New file |
| `src/react/` | All React components, hooks, contexts | New directory |
| `src/gen/` | Protobuf-generated clients | Shared — no changes needed |
| `src/infrastructure/` | ConnectRPC calls, WebSocket, telemetry | Shared — import directly from React |
| `webserver/web/templates/index-lit.html` | Lit SPA shell | Rename from `index.html` |
| `webserver/web/templates/index-react.html` | React SPA shell | New file |
| `statichttp/handler.go` | Route `/lit/*` and `/react/*` to correct template | Modified |
| `statichttp/asset_handler.go` | Return correct script tags per route | Modified |

## Recommended Project Structure

```
webserver/web/
├── src/
│   ├── main.ts                    # Lit entry (existing, unchanged)
│   ├── react-main.tsx             # React entry (NEW)
│   │
│   ├── react/                     # React app (NEW directory)
│   │   ├── App.tsx                # Root component, router setup
│   │   ├── contexts/              # React Context providers
│   │   │   ├── ProcessingContext.tsx
│   │   │   ├── FilterContext.tsx
│   │   │   └── AppContext.tsx
│   │   ├── hooks/                 # Custom hooks
│   │   │   ├── useImageProcessing.ts
│   │   │   ├── useVideoStream.ts
│   │   │   ├── useFilters.ts
│   │   │   ├── useFileUpload.ts
│   │   │   └── useHealthMonitor.ts
│   │   ├── components/            # React components
│   │   │   ├── layout/            # Navbar, Sidebar, etc.
│   │   │   ├── video/             # CameraPreview, VideoGrid, etc.
│   │   │   ├── image/             # ImageSelector, ImageProcessor
│   │   │   ├── filters/           # FilterPanel, FilterControl
│   │   │   ├── files/             # FileUpload, VideoUpload
│   │   │   ├── settings/          # ConfigPanel, FeatureFlags
│   │   │   └── ui/                # Toast, Modal, FAB, etc.
│   │   └── types/                 # React-specific TypeScript types
│   │
│   ├── gen/                       # Protobuf generated (SHARED — no changes)
│   ├── infrastructure/            # Services (SHARED — import directly)
│   │   ├── connection/
│   │   ├── data/
│   │   ├── external/
│   │   ├── observability/
│   │   └── transport/
│   ├── application/               # DI container (Lit-specific — do NOT share)
│   ├── components/                # Lit Web Components (existing, unchanged)
│   ├── domain/                    # Domain interfaces (SHARED)
│   └── services/                  # Business logic (SHARED — import directly)
│
├── templates/
│   ├── index-lit.html             # Renamed from index.html
│   └── index-react.html           # New React SPA shell
│
├── vite.config.ts                 # Modified for dual entry
├── package.json                   # Add React deps
└── static/js/dist/
    ├── lit/                       # Lit build output
    └── react/                     # React build output
```

### Structure Rationale

- **`src/react/` as sibling to `src/components/`:** Keeps Lit and React code isolated with clear boundaries. Neither touches the other.
- **Shared `src/gen/`:** The protobuf-generated ConnectRPC clients are framework-agnostic TypeScript — React imports them exactly as Lit does. Zero duplication.
- **Shared `src/infrastructure/`:** Services like `grpc-frame-transport.ts`, `video-service.ts`, `accelerator-health-monitor.ts` are plain TypeScript classes with no Lit dependency. React hooks wrap them without copying them.
- **Separate `src/application/DI` is NOT shared:** The existing DI container uses Lit's singleton services with side effects. React uses Context instead of a DI container.
- **Dual dist output dirs:** `static/js/dist/lit/` and `static/js/dist/react/` prevent filename collisions between builds.

## Architectural Patterns

### Pattern 1: Vite MPA with Two HTML Entry Points

**What:** Configure `rollupOptions.input` with two keys — each pointing to a separate HTML file. Each HTML file references its own JS entry. Vite builds them independently with code-splitting across both.

**When to use:** When two SPAs must coexist in one Vite project with shared chunks.

**Trade-offs:** Simple, no plugins required. Dev server serves both. Shared vendor chunks (React, Lit) are deduplicated. The downside: both apps build together — no independent build. Acceptable here since they share the same project.

**Example:**
```typescript
// vite.config.ts — modified build section
build: {
  outDir: 'static/js/dist',
  emptyOutDir: true,
  sourcemap: true,
  manifest: true,
  rollupOptions: {
    input: {
      lit:   resolve(__dirname, 'templates/index-lit.html'),
      react: resolve(__dirname, 'templates/index-react.html'),
    },
    output: {
      entryFileNames: '[name]/app.[hash].js',
      chunkFileNames: 'shared/[name].[hash].js',
      assetFileNames: 'assets/[name].[hash][extname]',
    },
  },
},
```

Note: The Vite dev server in MPA mode automatically serves both HTML files at their respective paths (e.g., `/templates/index-lit.html`). The Go dev proxy must forward `/lit/*` and `/react/*` to Vite correctly.

### Pattern 2: React Hooks Wrapping Existing Infrastructure Services

**What:** Custom hooks import the existing infrastructure services (plain TypeScript singletons) and expose their state/actions as React state. The service is not modified — the hook adapts it.

**When to use:** Everywhere in the React app where business logic currently lives in `src/infrastructure/` or `src/services/`.

**Trade-offs:** Existing services are battle-tested and tested. No duplication. Only risk: if a service manages its own internal subscriptions (e.g., health monitor), the hook must wire up and tear down subscriptions in `useEffect`.

**Example:**
```typescript
// src/react/hooks/useHealthMonitor.ts
import { acceleratorHealthMonitor } from '../../infrastructure/external/accelerator-health-monitor';
import { useState, useEffect } from 'react';

export function useHealthMonitor() {
  const [healthy, setHealthy] = useState(true);

  useEffect(() => {
    acceleratorHealthMonitor.startMonitoring(
      () => setHealthy(false),
      () => false  // no modal check needed in React — use state instead
    );
    return () => acceleratorHealthMonitor.stopMonitoring();
  }, []);

  return { healthy };
}
```

### Pattern 3: React Context for Global State (No Prop Drilling)

**What:** Use React Context at the app root to hold services and shared state that multiple components need. Context replaces the Lit DI container and the pattern of passing services down as element properties.

**When to use:** For services that are needed across more than 2 component layers: config, filters, health status, WebRTC sessions.

**Trade-offs:** Simpler than Redux/Zustand for this scale. Context re-renders all consumers on value change — use `useMemo` or split contexts by update frequency to avoid over-rendering.

**Example:**
```typescript
// src/react/contexts/AppContext.tsx
const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [filters, setFilters] = useState<Filter[]>([]);
  const config = useMemo(() => streamConfigService, []);

  return (
    <AppContext.Provider value={{ filters, setFilters, config }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be inside AppProvider');
  return ctx;
}
```

### Pattern 4: Go Server Dual-Template Routing

**What:** Go's `statichttp.StaticHandler` is modified to serve different HTML templates based on URL prefix. `/lit/*` gets the Lit template, `/react/*` gets the React template. The catch-all `/` continues serving the Lit template for backward compatibility.

**When to use:** This is the integration seam — required for the dual-route approach.

**Trade-offs:** Minimal Go changes. The `ServeIndex` function already reads the template at request time in hot-reload mode, so the path just needs to be parameterized by route prefix. In production mode, two `*template.Template` instances are pre-parsed at startup.

## Data Flow

### Image Processing Request Flow (React)

```
User clicks "Process"
    ↓
<ImageProcessorPanel> component
    ↓
useImageProcessing() hook
    ↓
inputSourceService.processImage()   ← existing infrastructure service, unchanged
    ↓
ConnectRPC client (src/gen/image_processor_service_connect.ts)
    ↓
POST /cuda_learning.ImageProcessorService/ProcessImage
    ↓
Go ConnectRPC handler → C++ gRPC → CUDA kernel
    ↓
Protobuf response
    ↓
hook updates React state
    ↓
Component re-renders with processed image
```

### Video Streaming Flow (React)

```
User starts stream
    ↓
useVideoStream() hook
    ↓
webrtcService.createSession()   ← existing src/infrastructure/connection/webrtc-service.ts
    ↓
WebSocket /ws/webrtc-signaling  (SDP offer/answer, ICE)
    ↓
WebRTC data channel established
    ↓
Frames arrive → hook updates React state → <CameraPreview> re-renders
```

### State Management

```
AppContext (global)
  ├── filters state       ← FilterContext (update on user interaction)
  ├── config service ref  ← stable, no re-render
  └── health status       ← useHealthMonitor hook

Component local state
  ├── loading, error      ← each hook manages its own async state
  ├── form inputs         ← useState in form components
  └── modal open/close    ← useState in modal components
```

## Integration Points

### Vite Dev Server — Go Proxy Configuration

The existing `DevelopmentAssetHandler` proxies specific path prefixes from Go to Vite. For MPA dev mode, the config paths must include both app prefixes.

**Current config defaults** (`config.go`):
```
"server.dev_server_paths": ["/@vite/", "/src/", "/node_modules/"]
```

**Required additions:**
```
"server.dev_server_paths": ["/@vite/", "/src/", "/node_modules/", "/templates/"]
```

Vite dev server in MPA mode serves the HTML files at `/templates/index-lit.html` and `/templates/index-react.html`. The Go proxy must forward requests to those paths to Vite.

Alternative (cleaner): In dev mode, Go serves the HTML templates itself (already does this via `template.ParseFiles`), and only asset paths (`/src/`, `/@vite/`) are proxied to Vite. This is how the existing system already works — Vite's HMR script tag is injected by `DevelopmentAssetHandler.GetScriptTags()`. The React handler does the same with `react-main.tsx` as the module src.

### Shared ConnectRPC Client — How React Uses It

The generated clients in `src/gen/` are plain TypeScript functions. React components import them through the infrastructure services, not directly:

```typescript
// React hooks call existing services — same as Lit
import { inputSourceService } from '../../infrastructure/data/input-source-service';
import { videoService } from '../../infrastructure/data/video-service';
```

The singleton services already hold the ConnectRPC transport client internally. React gets the same client instance Lit uses. No configuration needed.

**One caution:** The existing services use module-level singletons. If a service stores Lit component references or dispatches DOM events (custom elements), that code must be bypassed in React usage. Review each service's event dispatch patterns before wrapping in a hook.

### Go Static File Serving — Production

In production, Go reads from `webserver/web/static/js/dist/`. The manifest file (`manifest.json`) maps entry names to hashed filenames. The `ProductionAssetHandler` reads this manifest to generate `<script>` tags.

**Required change:** `asset_manifest.go` must support reading two manifest sections (or two manifest files) and returning the correct script tags based on which frontend is being served. The simplest approach: Vite produces a single `manifest.json` with both entry keys (`lit` and `react`). The handler looks up `lit` or `react` by entry name.

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| ConnectRPC backend | Shared via existing infrastructure services | No changes to services needed |
| WebSocket `/ws` | `webrtc-service.ts` singleton — wrap in `useVideoStream` hook | Services handle connection lifecycle |
| WebSocket `/ws/webrtc-signaling` | Same webrtcService | Tear down in useEffect cleanup |
| OpenTelemetry | `telemetry-service.ts` — initialize once in `react-main.tsx` | Same init pattern as Lit's `main.ts` |
| Health monitor | `accelerator-health-monitor.ts` — wrap in `useHealthMonitor` | startMonitoring/stopMonitoring in useEffect |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| React components ↔ infrastructure services | Direct import + custom hooks | Services are plain TS, no Lit coupling |
| React app ↔ Go backend | ConnectRPC (same endpoints as Lit) | Zero backend changes |
| Lit frontend ↔ React frontend | None — fully isolated SPAs | Only share build artifacts and Go server |
| React hooks ↔ React contexts | `useContext` — no prop drilling | Split contexts by update frequency |

## Anti-Patterns

### Anti-Pattern 1: Duplicating Infrastructure Services

**What people do:** Copy `video-service.ts` into `src/react/infrastructure/` and modify it for React.

**Why it's wrong:** Creates two implementations of the same ConnectRPC calls. When the proto changes (re-running `scripts/build/protos.sh`), only one copy gets updated. Bugs diverge silently.

**Do this instead:** Import the existing service from `../../infrastructure/data/video-service` in the React hook. The service is a plain TypeScript class with no Lit dependency — it works as-is.

### Anti-Pattern 2: One Vite Entry for Both Apps

**What people do:** Import both `main.ts` and `react-main.tsx` from a single entry file, or use conditional logic to switch between them.

**Why it's wrong:** Forces Lit and React to load in the same bundle. The browser downloads both frameworks (Lit + React) even if only one is used. HMR for one app breaks the other. Bundle size doubles unnecessarily.

**Do this instead:** Use Vite MPA mode with `rollupOptions.input` — two separate HTML files, two separate entry modules. Shared vendor code (if any) is code-split automatically by Rollup.

### Anti-Pattern 3: Using the Lit DI Container in React

**What people do:** Call `container.getVideoService()` from inside React components or hooks, importing the Lit DI container.

**Why it's wrong:** The DI container (`src/application/di/Container.ts`) is designed for Lit's initialization flow. It may have side effects tied to Lit's component lifecycle. It will also make it unclear whether the React app has any Lit dependency.

**Do this instead:** Import infrastructure singletons directly (`import { videoService } from '../../infrastructure/data/video-service'`). Create a React-specific initialization sequence in `react-main.tsx` that mirrors what `main.ts` does for Lit.

### Anti-Pattern 4: Global State in One Monolithic Context

**What people do:** Create a single `AppContext` that holds all state — filters, video sessions, config, health status, upload progress.

**Why it's wrong:** Every state update (e.g., a video frame arriving) re-renders every component subscribed to the context, even if they only care about filters.

**Do this instead:** Split contexts by update frequency and domain. `FilterContext` (changes on user action), `ProcessingContext` (changes per frame — keep out of context, use local state or refs), `ConfigContext` (stable, changes rarely), `HealthContext` (polling interval).

## Scaling Considerations

This is a single-user educational application. Scaling is not a concern. Architecture decisions should optimize for learning clarity and maintainability, not throughput.

| Scale | Architecture |
|-------|--------------|
| 1 user (current) | Single Go process, single Vite build, no CDN needed |
| 10 users | Same — Go handles concurrent WebRTC sessions already |
| Production | Static assets behind CDN, Go server unchanged |

## Sources

- Vite MPA documentation: https://vite.dev/guide/build (multi-page app section)
- Existing codebase: `webserver/web/vite.config.ts` (current single-entry config)
- Existing codebase: `webserver/pkg/interfaces/statichttp/` (Go routing and template serving)
- Existing codebase: `webserver/pkg/config/config.go` (dev_server_paths configuration)
- Existing codebase: `webserver/web/src/application/di/Container.ts` (existing DI — shows what NOT to reuse)
- Existing codebase: `webserver/web/src/infrastructure/` (shared services — shows what to reuse directly)

---
*Architecture research for: React frontend migration — CUDA media processing platform*
*Researched: 2026-04-12*
