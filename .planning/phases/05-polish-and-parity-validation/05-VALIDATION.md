---
phase: 05
slug: polish-and-parity-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-13
---

# Phase 05 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Vitest 1.6.1 + Playwright 1.56.1 |
| **Config file** | `front-end/vitest.config.ts`, `front-end/playwright.config.ts` |
| **Quick run command** | `cd front-end && npm run test -- --run` |
| **Full suite command** | `./scripts/test/unit-tests.sh --skip-golang --skip-cpp` then `./scripts/test/e2e.sh --chromium` (per `CLAUDE.md`) |
| **Estimated runtime** | ~2–15 minutes (depends on e2e scope) |

---

## Sampling Rate

- **After every task commit:** `cd front-end && npx tsc --noEmit` (once clean) + targeted Vitest for touched areas
- **After every plan wave:** `npm run build` in `front-end/` + Vitest full run + Playwright subset aligned with that wave
- **Before `/gsd-verify-work`:** Full suite green; PAR-01 checklist ready for sign-off
- **Max feedback latency:** 600 seconds (full e2e budget)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | PAR-01 | T-05-01 / — | N/A (build gate) | typecheck + build | `cd front-end && npx tsc --noEmit && npm run build` | ⬜ | ⬜ pending |
| TBD | 02 | 2 | PAR-01 | T-05-02 / — | N/A (style parity) | visual + report | Manual + grep/audit scripts as per plan | ⬜ | ⬜ pending |
| TBD | 03 | 3 | PAR-01 | T-05-03 / — | Safe rendering (no unsafe HTML) | checklist + e2e assist | Playwright + manual dual-route | ⬜ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] **`npx tsc --noEmit` clean** in `front-end/` (currently fails — vite config typing; fix as part of phase)
- [ ] **Console error harness** — shared Playwright helper for `/react` and `/lit` first-load console policy
- [ ] **Parity-oriented e2e** — extend beyond `dual-frontend-routes.spec.ts` shell smoke where plans specify

*Existing Vitest + Playwright infrastructure covers most automation; Wave 0 closes strict typing and harness gaps.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|---------------------|
| Feature parity across routes | PAR-01 | Requirement text mandates developer verification of identical functional outputs | Side-by-side `/react` vs `/lit` checklist; record gaps in phase report |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 600s for full gate
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
