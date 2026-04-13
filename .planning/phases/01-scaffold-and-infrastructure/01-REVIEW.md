---
phase: 01-scaffold-and-infrastructure
reviewed: 2026-04-13T03:16:39Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - front-end/vite.config.ts
  - front-end/scripts/verify-vite-dev-routes.sh
  - front-end/nginx.conf
  - front-end/playwright.config.ts
  - front-end/tests/e2e/dual-frontend-routes.spec.ts
  - front-end/src/react/webrtc-stub-import.test.tsx
  - front-end/src/infrastructure/connection/webrtc-service.ts
  - scripts/dev/start-frontend.sh
findings:
  critical: 0
  warning: 0
  info: 3
  total: 3
status: clean
---

# Phase 01: Code Review Report

**Reviewed:** 2026-04-13T03:16:39Z  
**Depth:** standard  
**Files Reviewed:** 8  
**Status:** clean  

## Summary

Review focused on dev/prod routing for `/react` and `/lit`, Vite proxy behavior, Nginx rewrites, shell hardening, and Playwright defaults. No open redirects or Host-header–style proxy abuse was found. Trailing-slash variants are handled in Vite middleware and Nginx (`location = /react/`, `/lit/`). Remaining notes are informational (Playwright `parseInt`, `ignoreHTTPSErrors`, `start-frontend.sh` `set -e` only).

## Critical Issues

_None._

## Warnings

_None._ (WR-01 trailing-slash routing was fixed in Vite middleware and Nginx `location = /react/` and `/lit/`.)

## Info

### IN-01: `start-frontend.sh` uses only `set -e`

**File:** `scripts/dev/start-frontend.sh:2`  

**Issue:** Unlike `verify-vite-dev-routes.sh`, this script does not use `set -u` or `pipefail`, so unset variables expand empty and pipeline failures may be missed.

**Fix:** Align with the verify script where appropriate: `set -euo pipefail` (and quote `"${1:-}"` if you adopt `-u` for the help branch).

### IN-02: `PLAYWRIGHT_WORKERS` parsed without validation

**File:** `front-end/playwright.config.ts:27`  

**Issue:** `parseInt(process.env.PLAYWRIGHT_WORKERS)` without radix or `NaN` handling. A non-numeric value yields `NaN`, which can cause confusing Playwright startup failures in CI.

**Fix:**

```ts
const workersEnv = process.env.PLAYWRIGHT_WORKERS;
const parsed = workersEnv ? parseInt(workersEnv, 10) : NaN;
const workers = Number.isFinite(parsed) ? parsed : (process.env.CI ? 4 : 25);
```

### IN-03: Verify script may not kill the full Vite process tree

**File:** `front-end/scripts/verify-vite-dev-routes.sh:26-27`  

**Issue:** Background `npm run dev` means `VITE_PID` is often the **npm** wrapper. On `EXIT`, killing that PID may leave a child `node` (Vite) running in edge cases.

**Fix:** Prefer `exec npm run dev` in a subshell with a process group and `kill -- -$PGID`, or use `pkill -P "$VITE_PID"` after kill, or start Vite directly (e.g. `npx vite`) so the PID is the server process.

---

_Reviewed: 2026-04-13T03:16:39Z_  
_Reviewer: Claude (gsd-code-reviewer)_  
_Depth: standard_
