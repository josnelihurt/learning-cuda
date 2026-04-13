# Phase 1: Scaffold and Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-12 (updated)
**Phase:** 01-scaffold-and-infrastructure
**Areas discussed:** React file layout, Vite MPA config, Go routing strategy, React app shell content, WebRTC test stubs, Dev server proxy config, Build validation scope

---

## React File Layout

| Option | Description | Selected |
|--------|-------------|----------|
| src-react/ sibling | New `webserver/web/src-react/` alongside existing `src/` | |
| src/react/ nested | New `webserver/web/src/react/` inside existing `src/` | ✓ |
| Separate webserver/web-react/ | Entirely new directory with own package.json | |

**User's choice:** src/react/ nested
**Notes:** User preferred keeping React inside the existing source tree

### Shared imports

| Option | Description | Selected |
|--------|-------------|----------|
| Import directly from ../ | React imports from `@/gen/` and `@/infrastructure/` | ✓ |
| Re-export via src/react/lib/ | Indirection layer for explicit dependency tracking | |

**User's choice:** Import directly from ../
**Notes:** Zero duplication, single source of truth

### Test layout

| Option | Description | Selected |
|--------|-------------|----------|
| Co-located | Tests next to source files | ✓ |
| Separate __tests__/ | All tests in dedicated directory | |

**User's choice:** Co-located
**Notes:** Matches existing Lit convention

---

## Vite MPA Configuration

| Option | Description | Selected |
|--------|-------------|----------|
| Single config, multi-page | One vite.config.ts with rollupOptions.input to two HTMLs | ✓ |
| Two separate Vite configs | Two build steps, full isolation | |

**User's choice:** Single config, multi-page

### tsconfig handling

| Option | Description | Selected |
|--------|-------------|----------|
| Keep shared tsconfig as-is | Decorator options harmless for React | |
| Split tsconfigs | tsconfig.lit.json + tsconfig.react.json | ✓ |

**User's choice:** Split tsconfigs
**Notes:** Explicit separation of concerns

### Output directories

| Option | Description | Selected |
|--------|-------------|----------|
| Same output directory | Both to `static/js/dist/` | ✓ |
| Separate subdirectories | Lit to `dist/lit/`, React to `dist/react/` | |

**User's choice:** Same output directory

---

## Go Routing Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Path-based routing | `/lit/` and `/react/` routes explicitly registered | ✓ |
| Query param/header routing | Single `/` route with frontend selector | |

**User's choice:** Path-based routing

### Root redirect

| Option | Description | Selected |
|--------|-------------|----------|
| Redirect to /react | React as default experience | ✓ |
| Redirect to /lit | Keep Lit as default until parity | |
| Landing page with links | Simple page linking to both | |

**User's choice:** Redirect to /react

### Template setup

| Option | Description | Selected |
|--------|-------------|----------|
| Separate template file | New `templates/react.html` | ✓ |
| Shared template with conditional | Single template with route-based script injection | |

**User's choice:** Separate template file

---

## React App Shell Content

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal shell with navbar | Header matching Lit branding + placeholder body | ✓ |
| Bare minimum ("Hello React") | Just text proving pipeline works | |
| Skeleton layout | Full layout with empty placeholders | |

**User's choice:** Minimal shell with navbar

### CSS strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse existing main.css | Import same CSS Lit uses | ✓ |
| New React-specific CSS | Start fresh for React | |

**User's choice:** Reuse existing main.css

---

## WebRTC Test Stubs

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal global stubs | `vi.fn()` stubs in existing test-setup.ts | ✓ |
| Typed mock objects | Full mock classes matching WebRTC API | |
| Conditional stubs | Only in React-specific test setup file | |

**User's choice:** Minimal global stubs
**Notes:** Phase 4 adds real WebRTC testing

---

## Dev Server Proxy Config

| Option | Description | Selected |
|--------|-------------|----------|
| Go proxies all non-API to Vite | No proxy changes needed, Vite MPA handles both entries | ✓ |
| Register explicit /react/ proxy | Add proxy route for React in Go | |
| You decide | Let researcher/planner determine strategy | |

**User's choice:** Go proxies all non-API to Vite

---

## Build Validation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| All 4 criteria are blockers | Full validation required | ✓ |
| Dev mode enough, prod soft | Faster iteration, fix prod issues in Phase 2 | |

**User's choice:** All 4 criteria are blockers
**Notes:** Foundation phase — everything else builds on this

---

## Context Update Review (2026-04-12)

| Option | Description | Selected |
|--------|-------------|----------|
| Vite MPA approach | D-04 revision: current config uses TS entry, not HTML. Revisit the approach? | |
| Go handler multi-entry | Dev mode script tags and template parsing hard-coded for single entry. How should dual-route serving work? | |
| Context looks good | Existing decisions hold. Nuances are the agent's discretion. No changes needed. | ✓ |

**User's choice:** Context looks good — existing 17 decisions verified against codebase, no revisions needed
**Notes:** Codebase scout confirmed all decisions valid. Added evidence for agent's discretion items: manifest key change (`asset_manifest.go:21`), dev handler hard-coded tags (`development_handler.go:47-52`), template parsing strategy (`handler.go:39`).

---

## the agent's Discretion

- Exact `asset_manifest.go` refactoring to support multiple manifest keys
- Multi-entry dev handler: `DevelopmentAssetHandler.GetScriptTags()` currently returns hard-coded script tags — needs route-aware selection
- Template parsing strategy: `NewStaticHandler` parses single template at construction — needs dual-template support
- React HTML template boilerplate (meta tags, font imports, etc.)
- Error states for `/react` route (e.g., build missing)
- How to structure `tsconfig.base.json` shared settings

## Deferred Ideas

None — discussion stayed within phase scope
