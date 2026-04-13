---
phase: 01-scaffold-and-infrastructure
plan: 01
subsystem: infra
tags: vite, react, typescript, multi-page, build-system

# Dependency graph
requires:
  - phase: None
    provides: Existing Lit frontend, Go backend, Vite build system
provides:
  - Multi-page Vite configuration with dual entry points (lit + react)
  - Split TypeScript configs (base, Lit-specific, React-specific)
  - React app shell with navbar matching Lit branding
  - WebRTC API stubs for React test imports
  - Build produces both lit.[hash].js and react.[hash].js bundles
affects: 02-react-hooks, 03-react-components, 04-webrtc-integration

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Multi-page Vite config with named entry points
    - TypeScript config inheritance (base + specific configs)
    - WebRTC global stubs for test isolation
    - React 18 createRoot API
    - Go template pattern shared across Lit and React HTML

key-files:
  created:
    - webserver/web/tsconfig.base.json - Shared TypeScript configuration
    - webserver/web/tsconfig.react.json - React-specific TypeScript config
    - webserver/web/templates/react.html - React HTML template with Go template syntax
    - webserver/web/src/react/main.tsx - React entry point
    - webserver/web/src/react/App.tsx - Minimal React app shell
  modified:
    - webserver/web/vite.config.ts - Updated for dual TypeScript entry points
    - webserver/web/tsconfig.json - Now extends tsconfig.base.json
    - webserver/web/src/test-setup.ts - Added WebRTC API stubs
    - webserver/web/vitest.config.ts - Updated to include .tsx test files

key-decisions:
  - "Vite uses TypeScript entry points, not HTML files (HTML templates contain Go syntax that Vite cannot parse)"
  - "TypeScript configs split into base (shared) + specific (Lit/React) to avoid decorator transform conflicts"
  - "WebRTC stubs added to test-setup.ts as global mocks (will be replaced with typed mocks in Phase 4)"
  - "React app shell uses same CSS classes as Lit for consistent branding"

patterns-established:
  - "Pattern 1: Multi-page Vite config uses named entry points producing [name].[hash].js output"
  - "Pattern 2: TypeScript config inheritance via 'extends' field for shared compiler options"
  - "Pattern 3: HTML templates use Go template syntax ({{range .ScriptTags}}) for script injection"
  - "Pattern 4: Test stubs for browser APIs (WebRTC) added globally via test-setup.ts"

requirements-completed: [SCAF-02]

# Metrics
duration: 13 min
completed: 2026-04-13
---

# Phase 01: Plan 01 Summary

**Multi-page Vite build with dual Lit/React entry points, split TypeScript configs, React app shell, and WebRTC test stubs**

## Performance

- **Duration:** 13 min (835 seconds)
- **Started:** 2026-04-13T01:12:34Z
- **Completed:** 2026-04-13T01:26:20Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Configured Vite as multi-page app with dual TypeScript entry points (lit + react)
- Split TypeScript configs into base (shared) + specific (Lit/React) configurations
- Created React HTML template with Go template syntax matching Lit template
- Created React app shell with navbar matching Lit branding and "React" badge
- Added WebRTC API stubs to test-setup.ts for React test imports
- Build produces both lit.[hash].js (419KB) and react.[hash].js (193KB) bundles
- All existing Lit tests pass without modification (178 tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Configure Vite MPA, split tsconfigs, install React deps, add WebRTC stubs** - `33c9b1f` (feat)
2. **Task 2: Create React HTML template, entry point, and minimal App shell** - `48c1305` (feat)

**Plan metadata:** [to be committed by orchestrator] (docs: complete plan)

## Files Created/Modified

- `webserver/web/vite.config.ts` - Updated with dual TypeScript entry points (src/main.ts, src/react/main.tsx)
- `webserver/web/tsconfig.base.json` - Shared TypeScript config without decorator settings
- `webserver/web/tsconfig.json` - Now extends tsconfig.base.json with Lit-specific decorator settings
- `webserver/web/tsconfig.react.json` - React-specific config extending base with jsx: "react-jsx"
- `webserver/web/src/test-setup.ts` - Added WebRTC API stubs (RTCPeerConnection, navigator.mediaDevices)
- `webserver/web/vitest.config.ts` - Updated include patterns to support .tsx test files
- `webserver/web/templates/react.html` - React HTML template with Go template syntax
- `webserver/web/src/react/main.tsx` - React 18 entry point using createRoot API
- `webserver/web/src/react/App.tsx` - Minimal React app shell with navbar matching Lit branding

## Decisions Made

- **Vite entry points must be TypeScript files, not HTML templates**: The original plan specified HTML entry points, but HTML templates contain Go template syntax (`{{range .ScriptTags}}`) that Vite cannot parse. Fixed by using TypeScript entry points instead.
- **TypeScript configs split via inheritance**: Base config contains shared options (target, moduleResolution, etc.) while specific configs add Lit (decorators) or React (jsx) settings.
- **React app uses same CSS as Lit**: Reuses `/static/css/main.css` and same CSS class names (navbar, navbar-brand, accent) for consistent branding.
- **No DI container in React entry**: Per plan D-14, React main.tsx does not import from DI container (unlike Lit main.ts which loads `application/di`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed vite.config.ts entry point configuration**
- **Found during:** Task 1 (Build verification)
- **Issue:** Plan specified HTML file entry points (`templates/index.html`, `templates/react.html`), but Vite failed with parse5 error: "missing-whitespace-between-attributes" at Go template syntax `{{if .Module}}`. Vite cannot parse HTML files containing Go template syntax.
- **Fix:** Changed vite.config.ts rollupOptions.input from HTML files to TypeScript files:
  - `lit: resolve(__dirname, 'src/main.ts')`
  - `react: resolve(__dirname, 'src/react/main.tsx')`
- **Files modified:** webserver/web/vite.config.ts
- **Verification:** Build succeeds, produces both lit.[hash].js and react.[hash].js bundles
- **Committed in:** 33c9b1f (Task 1 commit)

**2. [Rule 3 - Blocking] Created minimal React stubs to allow build to pass**
- **Found during:** Task 1 (Build verification)
- **Issue:** After fixing vite.config.ts to use TypeScript entry points, build failed because `src/react/main.tsx` and `src/react/App.tsx` did not exist (planned for Task 2). Build required both entry points to exist.
- **Fix:** Created minimal stub files that were replaced by proper implementation in Task 2:
  - `src/react/main.tsx`: Console.log stub
  - `src/react/App.tsx`: Null-returning component stub
- **Files modified:** webserver/web/src/react/main.tsx, webserver/web/src/react/App.tsx (created, then replaced in Task 2)
- **Verification:** Build passed after stubs created, final build passes with full React implementation
- **Committed in:** 33c9b1f (Task 1 commit) and 48c1305 (Task 2 commit)

**3. [Rule 3 - Blocking] Fixed plan sequencing issue**
- **Found during:** Task 1 (Acceptance criteria verification)
- **Issue:** Task 1 acceptance criteria required `npm run build` to pass, but Task 2 creates the actual React source files. This created a circular dependency.
- **Fix:** Created minimal stubs in Task 1 to satisfy build requirement, replaced with full implementation in Task 2.
- **Files modified:** webserver/web/src/react/main.tsx, webserver/web/src/react/App.tsx
- **Verification:** Build passes in both Task 1 (with stubs) and Task 2 (with full implementation)
- **Committed in:** 33c9b1f and 48c1305

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** All auto-fixes necessary for correctness. The HTML entry point issue was a fundamental misunderstanding in the plan - HTML templates with Go syntax cannot be Vite entry points. The stub creation was required to satisfy the plan's acceptance criteria sequencing.

## Issues Encountered

- **Vite HTML entry point parse error**: Plan specified HTML file entry points, but Go template syntax (`{{if .Module}}`) broke Vite's HTML5 parser. Fixed by using TypeScript entry points instead, which is the correct approach for this architecture where Go templates are processed server-side.
- **Task sequencing circular dependency**: Task 1 required build to pass, but Task 2 creates the React files. Worked around by creating minimal stubs in Task 1 that were replaced in Task 2.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Vite multi-page configuration complete and verified
- Both Lit and React entry points build successfully
- Manifest.json contains entries for both `src/main.ts` (lit) and `src/react/main.tsx` (react)
- React app shell is minimal (no feature logic, no DI container) - ready for Phase 2 hooks
- WebRTC stubs allow importing React modules that reference browser WebRTC APIs
- TypeScript configs properly separated to avoid decorator/JSX conflicts

**Ready for Phase 02 (React Hooks)**: Custom hooks can now be developed using the established React foundation.

## Self-Check: PASSED

- ✅ All created files exist: tsconfig.base.json, tsconfig.react.json, templates/react.html, src/react/main.tsx, src/react/App.tsx
- ✅ All modified files exist: vite.config.ts, tsconfig.json, test-setup.ts, vitest.config.ts
- ✅ All commits present: 33c9b1f (Task 1), 48c1305 (Task 2)
- ✅ SUMMARY.md created and populated
- ✅ Vite builds dual entry points: lit.[hash].js (419KB) and react.[hash].js (193KB)
- ✅ Manifest.json has 2 entries (lit and react)
- ✅ React App shell has navbar-brand class and "CUDA" + "Image Processor" text
- ✅ WebRTC stubs present in test-setup.ts
- ✅ All existing Lit tests pass (178 tests, 11 test files)

---
*Phase: 01-scaffold-and-infrastructure*
*Completed: 2026-04-13*
