---
phase: 01
slug: scaffold-and-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-12
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Vitest ^1.2.0 + happy-dom ^12.10.3 |
| **Config file** | `front-end/vitest.config.ts` |
| **Quick run command** | `cd front-end && npx vitest run --passWithNoTests` |
| **Full suite command** | `cd front-end && npx vitest run` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** `cd front-end && npx vitest run` (targeted if possible) + `npm run build` when touching Vite/React
- **After every plan wave:** `cd front-end && npx vitest run` + `npm run build`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | SCAF-02 | — | N/A static assets | build | `cd front-end && npm run build && test -f dist/index.html && test -f dist/react.html` | Wave 0 | pending |
| TBD | 01 | 1 | SCAF-01 | — | N/A route wiring | e2e/manual | Playwright or scripted HTTP per plan tasks | Wave 0 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] E2E or scripted checks for `/react` and `/lit` on the agreed dev/prod host (SCAF-01)
- [ ] Minimal React/Vitest coverage proving `test-setup.ts` WebRTC stubs allow imports

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Browser loads correct shell at `/react` and `/lit` | SCAF-01 | Path behavior depends on Nginx/Traefik/Vite dev matrix | Follow success criteria in ROADMAP for Phase 1 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency under cap
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
