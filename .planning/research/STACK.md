# Stack Research

**Domain:** React frontend migration — brownfield addition alongside existing Lit Web Components
**Researched:** 2026-04-12
**Confidence:** HIGH (versions verified via npm/official sources)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| react | ^19.2.5 | UI rendering | Latest stable; concurrent features, improved Suspense, no breaking changes from 18 for this use case |
| react-dom | ^19.2.5 | DOM renderer | Paired with React, required for browser rendering |
| react-router | ^7.11.0 | Client-side routing | v7 stable; `react-router-dom` is deprecated — import from `react-router` directly; enables `/lit` and `/react` route split cleanly |
| @vitejs/plugin-react | ^4.x | Vite JSX transform + HMR | Official plugin; no Babel dep in v4 for basic use; drop-in addition to existing vite.config.ts |

Note on `@vitejs/plugin-react` version: v6 is paired with Vite 8. The existing project uses Vite 5. Use v4 (latest compatible with Vite 5). If/when Vite is upgraded to 8, upgrade plugin to v6 together.

### State Management Decision

**Recommendation: React Context for this project — do NOT add Zustand yet.**

Rationale: The PROJECT.md explicitly lists "Context-based state management" as a learning goal. Adding Zustand bypasses the learning objective. Context is the right primitive for this codebase size (parity with Lit's DI container patterns, not a large SPA with heavy cross-cutting state). The video streaming and image processing state is largely local to feature components. Zustand adds value at scale; this project benefits from the constraint of learning Context patterns first.

If re-renders become a demonstrated problem (measured, not assumed), introduce Zustand at that point. React Context + `useMemo`/`useCallback` handles the load.

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @testing-library/react | ^16.3.2 | Component testing | All React component tests; replaces @open-wc/testing-helpers |
| @testing-library/dom | ^10.x | Peer dep of RTL v16 | Required alongside @testing-library/react v16+ |
| @testing-library/user-event | ^14.x | User interaction simulation | For simulating clicks, input, file upload in tests |
| @types/react | ^19.x | TypeScript types | Required for TypeScript strict mode in JSX |
| @types/react-dom | ^19.x | TypeScript types | Required for React DOM TypeScript integration |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| eslint-plugin-react | ESLint rules for React | Add to existing ESLint config; disable `react/react-in-jsx-scope` (React 19 auto-import) |
| eslint-plugin-react-hooks | Enforces hooks rules | Critical: catches hook ordering bugs at lint time, not runtime |
| @typescript-eslint (existing) | Already present | No change needed; already handles TSX if `.tsx` extension added to lint glob |

## Installation

```bash
# Core React
npm install react@^19.2.5 react-dom@^19.2.5 react-router@^7.11.0

# TypeScript types (dev)
npm install -D @types/react@^19 @types/react-dom@^19

# Vite React plugin (dev) — use v4 to stay compatible with Vite 5
npm install -D @vitejs/plugin-react@^4

# Testing (dev)
npm install -D @testing-library/react@^16.3.2 @testing-library/dom@^10 @testing-library/user-event@^14

# ESLint React plugins (dev)
npm install -D eslint-plugin-react eslint-plugin-react-hooks
```

## Vite Config Integration

The existing `vite.config.ts` needs one addition — import and register the React plugin. Everything else (proxy, HMR, HTTPS, aliases) stays unchanged:

```typescript
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [
    gitVersionPlugin(),
    react(),   // Add this — handles JSX transform for .tsx files
  ],
  // ... rest unchanged
});
```

The existing `rollupOptions.input` points to `src/main.ts` (Lit entry). For dual-entry (Lit + React), Rollup input must be changed to an object:

```typescript
rollupOptions: {
  input: {
    lit: resolve(__dirname, 'src/main.ts'),
    react: resolve(__dirname, 'src/react-main.tsx'),
  },
  // ...
}
```

Go's static file server (`statichttp/`) will need to serve both `index.html` (Lit) and a `react.html` (React) based on route prefix. This is a Go-side routing concern, not a Vite concern.

## Routing Strategy

React Router v7 with `createBrowserRouter` scoped to `/react/*` prefix. The Lit frontend keeps its existing routes. The Go server serves `react.html` for `/react/*` requests and `index.html` for everything else.

Within the React app, routes map to feature parity:
- `/react/` — dashboard/home
- `/react/images` — image processing
- `/react/video` — video streaming
- `/react/settings` — configuration

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| React Context | Zustand 5 | When measured re-render problems appear; or if state crosses many unrelated component trees |
| React Context | Redux Toolkit | Never for this project — Redux is for teams with complex state machines and time-travel debugging needs; overkill here |
| react-router v7 | TanStack Router | If you need type-safe route params end-to-end and file-based routing; not needed for a feature-parity migration |
| @vitejs/plugin-react | @vitejs/plugin-react-swc | SWC plugin is faster in large projects; at this project's scale Babel-less v4 is fast enough; SWC adds complexity |

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Next.js | SSR framework would fight the existing Go server serving static files; no SSR needed here | Plain React + Vite (already in project) |
| Redux / Redux Toolkit | Heavyweight for a learning project with moderate state; adds boilerplate that obscures React patterns | React Context + custom hooks |
| Zustand (now) | Circumvents the learning goal; Context is sufficient and was explicitly chosen in PROJECT.md | React Context; revisit after Context patterns are mastered |
| react-query / TanStack Query | All data fetching goes through ConnectRPC hooks, not REST fetch calls; query caching adds a layer with no benefit here | ConnectRPC client directly in custom hooks |
| Storybook | Not in scope for this milestone; adds build complexity | Skip for parity milestone |
| @testing-library/react-hooks | Deprecated — renderHook is now part of @testing-library/react v14+ | @testing-library/react (already recommended) |
| react-dom/client (directly) | Use `createRoot` via react-dom; do not mix with Lit's custom element registration in same DOM tree | Separate HTML entry points (react.html vs index.html) |
| Emotion / styled-components | CSS-in-JS adds bundle weight; existing CSS will be reused directly | Reuse existing CSS files, plain className |

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| react@^19.2.5 | @vitejs/plugin-react@^4 | Compatible; v4 supports React 17+ JSX transform |
| react@^19.2.5 | @testing-library/react@^16.3.2 | Compatible; RTL 16 added React 19 support |
| react@^19.2.5 | react-router@^7.11.0 | Compatible; React Router 7 targets React 18+ |
| react@^19.2.5 | vitest@^1.2.0 (existing) | Compatible; RTL uses jsdom/happy-dom via vitest config |
| @vitejs/plugin-react@^4 | vite@^5.0.10 (existing) | Compatible; v4 is the Vite 5 generation plugin |
| @testing-library/react@^16 | @testing-library/dom@^10 | Required peer dep; must install both |

## Sources

- https://react.dev/blog/2024/12/05/react-19 — React 19 stable release (HIGH confidence)
- https://www.npmjs.com/package/react — React 19.2.5 latest version confirmed (HIGH confidence)
- https://reactrouter.com/changelog — React Router v7 stable, `react-router-dom` deprecated (HIGH confidence)
- https://www.npmjs.com/package/zustand — Zustand 5.0.12 latest stable (HIGH confidence)
- https://www.npmjs.com/package/@testing-library/react — RTL 16.3.2 latest (HIGH confidence)
- https://github.com/vitejs/vite-plugin-react — @vitejs/plugin-react v4/v6 release history (HIGH confidence)
- https://dev.to/hijazi313/state-management-in-2025-when-to-use-context-redux-zustand-or-jotai-2d2k — State management trade-off analysis (MEDIUM confidence, cross-referenced with multiple sources)

---
*Stack research for: React frontend migration of CUDA image/video processing platform*
*Researched: 2026-04-12*
