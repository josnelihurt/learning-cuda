---
phase: 2
slug: core-hook-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-12
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Vitest ^1.2.0 |
| **Config file** | `front-end/vitest.config.ts` |
| **Quick run command** | `cd front-end && npm run test` |
| **Full suite command** | `cd front-end && npx vitest run` |
| **Estimated runtime** | ~30–120 seconds (grows with new React hook tests) |

---

## Sampling Rate

- **After every task commit:** `cd front-end && npx vitest run` scoped to touched tests under `src/react/`
- **After every plan wave:** `cd front-end && npx vitest run`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | HOOK-01 | T-02-01 / — | Same-origin transport; clients only from context | unit | `cd front-end && npx vitest run src/react` | with plan 01 | pending |
| TBD | 01 | 1 | HOOK-02 | — | Unary RPC via PromiseClient | unit | `cd front-end && npx vitest run src/react` | with plan 01 | pending |
| TBD | 01 | 1 | HOOK-03 | — | Toast bridge; DOM host on react.html | unit | `cd front-end && npx vitest run src/react` | with plan 01 | pending |
| TBD | 02 | 2 | HOOK-04 | T-02-05 / — | ListFilters only via generated client | unit | `cd front-end && npx vitest run src/react` | with plan 02 | pending |
| TBD | 02 | 2 | HOOK-05 | — | Health RPC only; interval cleanup | unit | `cd front-end && npx vitest run src/react` | with plan 02 | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

Tests and `renderWithGrpcProviders` (or equivalent) are **created inside** `02-01-PLAN.md` and `02-02-PLAN.md` tasks — no separate Wave 0 plan. Before sign-off, all listed automated commands must pass.

- [ ] `front-end/src/react/**/*.test.tsx` — coverage for HOOK-01 through HOOK-05 (delivered by phase plans)
- [ ] Shared test helper for context-wrapped renders (delivered by plan 01 tasks)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Toast visible on `/react` | HOOK-03 | Web component + styling | Dev stack: open `/react`, trigger `useToast`, confirm toast in DOM |
| Health flips within one poll | HOOK-05 | Real backend timing | Stop/start Go service; observe status within 15–30s interval |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
