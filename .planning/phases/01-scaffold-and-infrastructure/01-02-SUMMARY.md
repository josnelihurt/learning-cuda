---
phase: 01-scaffold-and-infrastructure
plan: 02
subsystem: infra
tags: nginx, playwright, docker, scaf-01

requires:
  - phase: 01-01
    provides: Vite pretty paths and preview HTTP for Playwright
provides:
  - Nginx `location = /lit` mirroring `/react`
  - Playwright spec against vite preview for /react and /lit shells
  - start-frontend.sh help and runtime echo for SCAF-01 topology
affects: 02-react-hooks

tech-stack:
  added: []
  patterns:
    - Playwright webServer when PLAYWRIGHT_BASE_URL unset (dotenv no longer forces :3000)

key-files:
  created:
    - front-end/tests/e2e/dual-frontend-routes.spec.ts
  modified:
    - front-end/nginx.conf
    - front-end/playwright.config.ts
    - front-end/.env.development
    - scripts/dev/start-frontend.sh

key-decisions:
  - "Removed PLAYWRIGHT_BASE_URL from .env.development so preview webServer works; e2e.sh still exports URL for full-stack runs"

patterns-established:
  - "Dual-frontend regression via Chromium on local preview"

requirements-completed: [SCAF-01]

duration: 20min
completed: 2026-04-12
---

# Phase 1 Plan 02: Nginx, Playwright, developer topology

**Outcome:** Production-style `/lit` exists in Nginx; Playwright proves React and Lit shells on `vite preview`; developers see dev vs Nginx vs Go roles when starting the frontend.

## Task Commits

1. **Tasks 1–2: Nginx /lit, Playwright webServer + spec, .env.development** — `502e028` (test)
2. **Task 3: start-frontend documentation** — `790909a` (docs)

## Verification

- `grep` checks on `nginx.conf` and `start-frontend.sh` per plan
- `env -u PLAYWRIGHT_BASE_URL npx playwright test tests/e2e/dual-frontend-routes.spec.ts --project=chromium`

## Self-Check: PASSED
