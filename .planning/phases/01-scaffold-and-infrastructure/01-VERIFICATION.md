---
phase: 01-scaffold-and-infrastructure
verified: 2026-04-12T18:57:00Z
status: gaps_found
score: 3/6 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Developer visits /react in the browser and gets a React app shell (not a 404 or the Lit page)"
    status: partial
    reason: "Production mode broken: asset_manifest.go looks for HTML-path keys (templates/react.html) but Vite manifest uses TypeScript entry point keys (src/react/main.tsx). Lookup fails and returns default 'app.js'. Dev mode works correctly."
    artifacts:
      - path: "webserver/pkg/interfaces/statichttp/asset_manifest.go"
        issue: "GetEntryFile() uses wrong manifest keys - expects 'templates/index.html' and 'templates/react.html', but Vite generates 'src/main.ts' and 'src/react/main.tsx'"
    missing:
      - "Update asset_manifest.go GetEntryFile() to use TypeScript entry point keys: 'src/main.ts' for lit, 'src/react/main.tsx' for react"
  - truth: "Developer visits /lit and gets the existing Lit frontend unchanged"
    status: partial
    reason: "Production mode broken: asset_manifest.go looks for HTML-path keys (templates/index.html) but Vite manifest uses TypeScript entry point keys (src/main.ts). Lookup fails and returns default 'app.js'. Dev mode works correctly."
    artifacts:
      - path: "webserver/pkg/interfaces/statichttp/asset_manifest.go"
        issue: "GetEntryFile() uses wrong manifest keys - expects 'templates/index.html', but Vite generates 'src/main.ts'"
    missing:
      - "Update asset_manifest.go GetEntryFile() to use TypeScript entry point key: 'src/main.ts' for lit"
  - truth: "In production mode, asset manifest serves correct hashed bundle for each route"
    status: failed
    reason: "Asset manifest lookup fails for both routes due to key mismatch. Both lit and react lookups return default 'app.js' instead of actual hashed bundles (lit.Cn0KsrHK.js, react.Cz8h8cJr.js)"
    artifacts:
      - path: "webserver/pkg/interfaces/statichttp/asset_manifest.go"
        issue: "Manifest key mismatch causes all production routes to serve wrong bundle"
      - path: "webserver/web/static/js/dist/.vite/manifest.json"
        issue: "Actual keys are 'src/main.ts' and 'src/react/main.tsx', not the HTML-path keys Go code expects"
    missing:
      - "Fix asset_manifest.go to use correct manifest keys matching Vite's actual output"
human_verification:
  - test: "Start dev server and verify /react serves React shell"
    expected: "React app loads with 'CUDA Image Processor' navbar and 'React' badge"
    why_human: "Requires running server and visual browser verification - cannot verify programmatically"
  - test: "Start dev server and verify /lit serves existing Lit frontend"
    expected: "Lit app loads unchanged from previous implementation"
    why_human: "Requires running server and visual browser verification - cannot verify programmatically"
  - test: "Start dev server and verify / redirects to /react"
    expected: "Browser redirects from / to /react"
    why_human: "Requires running server and browser to follow redirect - cannot verify programmatically"
---

# Phase 1: Scaffold and Infrastructure Verification Report

**Phase Goal:** Developer can load the React frontend at `/react` and the Lit frontend at `/lit` simultaneously from a single Vite build and Go server
**Verified:** 2026-04-12T18:57:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth   | Status     | Evidence       |
| --- | ------- | ---------- | -------------- |
| 1   | npm run build produces two bundles (lit.[hash].js and react.[hash].js) in static/js/dist/ | ✓ VERIFIED | Build output: lit.Cn0KsrHK.js (419KB), react.Cz8h8cJr.js (193KB) |
| 2   | npm run dev starts without errors and serves both HTML entry points | ? UNCERTAIN | Dev server not started - requires human verification |
| 3   | React App shell renders with CUDA Image Processor navbar matching Lit branding | ✓ VERIFIED | App.tsx contains className="navbar", className="navbar-brand", "CUDA Image Processor" text |
| 4   | WebRTC stubs in test-setup.ts allow importing modules that reference RTCPeerConnection | ✓ VERIFIED | test-setup.ts has global.RTCPeerConnection and navigator.mediaDevices stubs |
| 5   | Existing Lit build is unchanged — same entry, same output behavior | ✓ VERIFIED | Build still produces lit.[hash].js bundle, 178 Lit tests pass |
| 6   | Developer visits /react and gets the React app shell (not 404 or Lit page) | ✗ PARTIAL | Dev mode: ✓ (serves /src/react/main.tsx). Production: ✗ (manifest lookup fails, returns 'app.js') |
| 7   | Developer visits /lit and gets the existing Lit frontend unchanged | ✗ PARTIAL | Dev mode: ✓ (serves /src/main.ts). Production: ✗ (manifest lookup fails, returns 'app.js') |
| 8   | Developer visits / (root) and gets redirected to /react | ? UNCERTAIN | Code verified (app.go line 284 redirects to /react), but requires running server to test |
| 9   | In production mode, asset manifest serves correct hashed bundle for each route | ✗ FAILED | Manifest key mismatch: Go looks for 'templates/*.html', Vite generates 'src/**/*.ts' keys |
| 10  | In dev mode, Vite dev server serves correct entry point for each route | ✓ VERIFIED | development_handler.go returns /src/main.ts for lit, /src/react/main.tsx for react |
| 11  | All 4 success criteria from ROADMAP are satisfied | ✗ FAILED | 2 of 4 ROADMAP SCs fail in production mode (SC1, SC2) |

**Score:** 3/6 must-haves verified (6 partial/failed, 2 uncertain requiring human)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| webserver/web/vite.config.ts | MPA config with dual HTML inputs, @vitejs/plugin-react | ✓ VERIFIED | Has dual entry points (src/main.ts, src/react/main.tsx), react() plugin, entryFileNames: '[name].[hash].js' |
| webserver/web/tsconfig.base.json | Shared TypeScript settings | ✓ VERIFIED | Contains shared compilerOptions, excludes node_modules and dist |
| webserver/web/tsconfig.react.json | React-specific TypeScript config | ✓ VERIFIED | Extends base, has jsx: "react-jsx", includes src/react/**/*.tsx |
| webserver/web/templates/react.html | React HTML template with #root div | ✓ VERIFIED | Has `<div id="root">`, links /static/css/main.css, uses Go template syntax |
| webserver/web/src/react/main.tsx | React entry point | ✓ VERIFIED | Uses createRoot API, imports and renders App component |
| webserver/web/src/react/App.tsx | Minimal React app shell | ✓ VERIFIED | Has navbar with "CUDA Image Processor" branding, "React" badge |
| webserver/web/src/test-setup.ts | WebRTC global stubs | ✓ VERIFIED | Stubs RTCPeerConnection and navigator.mediaDevices globally |
| webserver/pkg/interfaces/statichttp/asset_handler.go | Route-aware GetScriptTags interface | ✓ VERIFIED | GetScriptTags(route string) signature added |
| webserver/pkg/interfaces/statichttp/asset_manifest.go | Multi-entry manifest lookup by route | ✗ STUB | GetEntryFile() uses wrong keys - expects 'templates/*.html', Vite generates 'src/**/*.ts' |
| webserver/pkg/interfaces/statichttp/production_handler.go | Route-aware production script tags | ⚠️ HOLLOW | Calls manifest.GetEntryFile(route), but manifest lookup fails due to key mismatch |
| webserver/pkg/interfaces/statichttp/development_handler.go | Route-aware dev script tags | ✓ VERIFIED | Returns correct entry points: /src/main.ts for lit, /src/react/main.tsx for react |
| webserver/pkg/interfaces/statichttp/handler.go | Dual-template serving with route selection | ✓ VERIFIED | Has litTmpl and reactTmpl fields, ServeIndex determines route from URL path |
| webserver/pkg/app/app.go | Catch-all route with /lit, /react, / redirect | ✓ VERIFIED | Root redirects to /react, other routes call serveIndex |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | --- | --- | ------ | ------- |
| webserver/pkg/app/app.go | webserver/pkg/interfaces/statichttp/handler.go | ServeIndex handler | ✓ WIRED | app.go line 289 calls serveIndex(w, r) |
| webserver/pkg/interfaces/statichttp/handler.go | webserver/pkg/interfaces/statichttp/asset_handler.go | GetScriptTags(route) call | ✓ WIRED | handler.go line 119 calls h.assetHandler.GetScriptTags(route) |
| webserver/pkg/interfaces/statichttp/production_handler.go | webserver/pkg/interfaces/statichttp/asset_manifest.go | manifest.GetEntryFile(route) | ⚠️ BROKEN | Production handler calls manifest.GetEntryFile(route), but it returns 'app.js' for both routes due to key mismatch |
| webserver/web/src/react/main.tsx | webserver/web/src/react/App.tsx | import App | ✓ WIRED | main.tsx imports App from './App' |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| webserver/pkg/interfaces/statichttp/production_handler.go | ScriptTags.Src | h.manifest.GetEntryFile(route) | ✗ NO | Returns static 'app.js' for both routes - manifest lookup fails |
| webserver/pkg/interfaces/statichttp/development_handler.go | ScriptTags.Src | entry variable (route-based) | ✓ YES | Returns real entry points: /src/main.ts or /src/react/main.tsx |
| webserver/web/src/react/App.tsx | Navbar text | Static JSX | ✓ YES | Contains "CUDA Image Processor" as static content |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Build produces dual bundles | cd webserver/web && npm run build | Built lit.Cn0KsrHK.js (419KB) and react.Cz8h8cJr.js (193KB) | ✓ PASS |
| Manifest has correct keys | cat webserver/web/static/js/dist/.vite/manifest.json | Keys are "src/main.ts" and "src/react/main.tsx" | ✓ PASS |
| Go code compiles | cd webserver && go build ./... | Compiled successfully | ✓ PASS |
| Go tests pass | cd webserver && go test ./pkg/... -count=1 | All 11 test packages passed | ✓ PASS |
| Frontend tests pass | cd webserver/web && npm test -- --run | 178 tests passed | ✓ PASS |
| Manifest lookup test | Custom Go test (see verification) | GetEntryFile("lit") returns "app.js" (WRONG - should be "lit.Cn0KsrHK.js") | ✗ FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| SCAF-01 | 01-02-PLAN.md | Developer can run React frontend at `/react` and Lit frontend at `/lit` from the same Go server simultaneously | ✗ PARTIAL | Dev mode: ✓ routes work correctly. Production: ✗ manifest lookup fails, both return 'app.js' |
| SCAF-02 | 01-01-PLAN.md | Vite is configured as a multi-page app (MPA) with separate entry points for Lit and React builds | ✓ VERIFIED | vite.config.ts has dual entry points, build produces both bundles |

**Orphaned Requirements:** None - both SCAF-01 and SCAF-02 are mapped to plans

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| None found | - | - | - | - |

### Human Verification Required

### 1. Dev Server Route Verification

**Test:** Start the dev server and navigate to /react, /lit, and /
**Expected:**
- /react loads React app with "CUDA Image Processor" navbar and "React" badge
- /lit loads existing Lit frontend unchanged
- / redirects to /react
**Why human:** Requires running server and visual browser verification - cannot verify programmatically

### 2. Production Build Route Verification (after fix)

**Test:** After fixing the manifest key mismatch, build production assets and verify routes
**Expected:**
- /react loads React app with correct hashed bundle (react.[hash].js)
- /lit loads Lit app with correct hashed bundle (lit.[hash].js)
**Why human:** Requires running production server and visual browser verification - cannot verify programmatically

### Gaps Summary

**Critical Bug: Production Asset Manifest Key Mismatch**

The Go server's asset manifest lookup is broken in production mode. The code expects HTML-path manifest keys (`templates/index.html`, `templates/react.html`), but Vite actually generates TypeScript entry point keys (`src/main.ts`, `src/react/main.tsx`).

**Impact:**
- Both /react and /lit routes in production mode will attempt to load "app.js" (the default fallback) instead of the correct hashed bundles
- Lit route: Expected to load `lit.Cn0KsrHK.js`, actually loads `app.js` (404)
- React route: Expected to load `react.Cz8h8cJr.js`, actually loads `app.js` (404)
- Development mode works correctly because it doesn't use the manifest lookup

**Root Cause:**
During plan 01-01 execution, a deviation occurred: Vite entry points were changed from HTML files to TypeScript files (because HTML templates contain Go syntax that Vite cannot parse). However, the Go code in plan 01-02 was not updated to reflect this change - it still expects the original HTML-path keys.

**Evidence:**
```bash
# Actual manifest keys:
$ cat webserver/web/static/js/dist/.vite/manifest.json
{
  "src/main.ts": { "file": "lit.Cn0KsrHK.js", ... },
  "src/react/main.tsx": { "file": "react.Cz8h8cJr.js", ... }
}

# What Go code looks for:
# asset_manifest.go line 21-24:
key := "templates/index.html"
if route == "react" {
    key = "templates/react.html"
}

# Test result:
# GetEntryFile("lit") returns "app.js" (WRONG)
# GetEntryFile("react") returns "app.js" (WRONG)
```

**Required Fix:**
Update `webserver/pkg/interfaces/statichttp/asset_manifest.go` to use the correct manifest keys:
```go
func (m ViteManifest) GetEntryFile(route string) string {
    key := "src/main.ts"
    if route == "react" {
        key = "src/react/main.tsx"
    }
    if entry, ok := m[key]; ok {
        return entry.File
    }
    return "app.js"
}
```

This is a straightforward one-line fix (change the key values) that will unblock production mode serving for both frontends.

---

_Verified: 2026-04-12T18:57:00Z_
_Verifier: the agent (gsd-verifier)_
