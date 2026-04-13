# Phase 05: Polish and Parity Validation - Research

**Researched:** 2026-04-13  
**Domain:** Frontend parity validation (Lit vs React), TypeScript strictness, CSS/visual alignment, manual QA process  
**Confidence:** MEDIUM (strong on toolchain facts; parity criteria are inherently subjective without automated visual baselines)

## Summary

Phase 5 closes the React migration loop by proving **functional and visual parity** between the Vite MPA routes `/lit` (Lit, `index.html` → `app-root`) and `/react` (React, `react.html` → `#root`), per **PAR-01** [VERIFIED: `.planning/REQUIREMENTS.md`, `front-end/vite.config.ts`, `front-end/tests/e2e/dual-frontend-routes.spec.ts`]. The roadmap adds non-functional gates: **TypeScript strict-mode cleanliness**, **no console errors on load**, and **CSS/layout parity** [VERIFIED: `.planning/ROADMAP.md` Phase 5 success criteria].

A critical planning fact: **`npm run build` (Vite) is not a substitute for TypeScript type checking.** Vite transpiles `.ts`/`.tsx` but does not run the TypeScript compiler for semantic validation [CITED: https://vite.dev/guide/features.html — TypeScript section]. The repo already sets `"strict": true` in `front-end/tsconfig.base.json` [VERIFIED: file read], but **explicit `tsc --noEmit` (or CI equivalent) is required** to claim “zero strict-mode errors.” As of this research run, `npx tsc --noEmit` in `front-end/` **exits with errors** (example: `vite.config.ts` `preview.https` type mismatch with `UserConfig`) [VERIFIED: command run 2026-04-13] — treat as a **pre-existing gap** for Phase 5 build-validation tasks.

Console cleanliness is best automated with **Playwright `page.on('console', …)`** filtering `msg.type() === 'error'` [CITED: Context7 `/microsoft/playwright` → `Page.on('console')` / `ConsoleMessage` docs]. The project already ships Playwright (`front-end/playwright.config.ts` starts `vite preview` on `127.0.0.1:4173` unless `PLAYWRIGHT_BASE_URL` is set) [VERIFIED: file read] and has a **dual-route smoke test** [VERIFIED: `front-end/tests/e2e/dual-frontend-routes.spec.ts`].

CSS parity should lean on **shared design tokens** (`var(--…)`) — React modules already reference custom properties with fallbacks [VERIFIED: grep `front-end/src/react/**/*.module.css`]. A manual **side-by-side checklist** (roadmap-aligned) remains the authoritative PAR-01 method; optional screenshot comparison can narrow drift but will not replace human judgment for “identical outputs” without baseline images.

**Primary recommendation:** Plan three tracks in lockstep — (1) **`tsc --noEmit` + fix** as the strict-mode gate, (2) **Playwright or manual protocol** for console + route smoke, (3) **token/CSS audit + side-by-side visual review** — document results in parity reports (artifact locations may follow existing draft plans under `.planning/phases/05-polish-parity/` [VERIFIED: glob] unless the milestone standardizes on this phase folder).

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PAR-01 | Developer can manually verify that `/react` and `/lit` routes produce functionally identical outputs for all features | Checklist-driven side-by-side testing; existing Playwright route smoke; feature list from REQUIREMENTS v1.0 (IMG/FILE/CONF/HLTH/VID) |
</phase_requirements>

## User Constraints

> **Note:** No phase `*-CONTEXT.md` was present. Constraints below are **locked by project requirements and roadmap** (not a discuss-phase transcript).

### Locked by REQUIREMENTS / ROADMAP

- **PAR-01** is the sole v1.0 requirement for Phase 5 [VERIFIED: `.planning/REQUIREMENTS.md`].
- Phase 5 **depends on Phase 4** (WebRTC/video) completing first [VERIFIED: `.planning/ROADMAP.md`].
- **Out of scope** for the product (must not expand Phase 5 into): backend changes, features beyond Lit parity, auth, v1.1 test-coverage mandates, Lit removal (CUT-*) [VERIFIED: `.planning/REQUIREMENTS.md` Out of Scope].

### Project Constraints (from .cursor/rules/)

No `.cursor/rules/` files were found in this workspace [VERIFIED: glob 2026-04-13].

## Standard Stack

### Core

| Library / tool | Version (this repo) | Purpose | Why Standard |
|----------------|---------------------|---------|--------------|
| TypeScript | 5.9.3 [VERIFIED: `npm ls typescript` in `front-end/`] | Strict type checking via `tsc` | Required to substantiate “strict mode” claims; Vite alone does not typecheck [CITED: vite.dev TypeScript section] |
| Vite | 5.4.20 [VERIFIED: `npm ls vite`] | MPA build (`index.html` + `react.html`), dev `/react` `/lit` rewrites | Already encodes dual entry and pretty routes [VERIFIED: `front-end/vite.config.ts`] |
| Vitest | 1.6.1 [VERIFIED: `npm ls vitest`] | Unit/component tests | Existing coverage gate config in `vitest.config.ts` [VERIFIED: file read] |
| Playwright (`@playwright/test`) | 1.56.1 [VERIFIED: `npm ls @playwright/test` / `npx playwright --version`] | E2E, console hooks, multi-browser | Already used for `dual-frontend-routes` [VERIFIED: `front-end/tests/e2e/`] |

### Supporting

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| ESLint + `@typescript-eslint/*` | ^6.19.0 (package.json) | Static analysis | Catches some issues `tsc` may not (style/import rules) [VERIFIED: `front-end/package.json`] |
| Prettier | ^3.2.4 | Formatting | Consistency across Lit/React TS [VERIFIED: `front-end/package.json`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual-only PAR-01 | Playwright + checklist | Automation reduces regressions but PAR-01 is **defined as manual** — automation should **support**, not replace, explicit human verification [VERIFIED: REQUIREMENTS wording] |
| `tsc` | `vite-plugin-checker` | Integrates typecheck into dev UX; adds dependency and config surface — only adopt if team wants inline DX vs CI `tsc` [ASSUMED] |

**Installation:** No new packages required for minimum parity validation; optional `vite-plugin-checker` only if chosen [ASSUMED].

**Version verification (registry latest vs repo):** `npm view` latest majors as of 2026-04-13: `typescript@6.0.2`, `vite@8.0.8`, `@playwright/test@1.59.1`, `vitest@4.1.4` [VERIFIED: npm registry]. The repo resolves older compatible versions — **plan tasks against installed `npm ls`**, not only `package.json` ranges.

## Architecture Patterns

### Recommended Project Structure

```
front-end/
├── index.html              # Lit shell
├── react.html              # React shell
├── vite.config.ts          # MPA inputs, /react /lit dev/preview middleware
├── tsconfig.base.json      # strict, skipLibCheck, paths @/*
├── src/
│   ├── components/         # Lit components + legacy CSS
│   └── react/              # React tree + CSS modules
└── tests/e2e/              # Playwright (incl. dual-frontend-routes)
```

### Pattern 1: Dual-route verification

**What:** For each feature, execute the same user journey on `/lit` and `/react` with the same backend state.  
**When to use:** PAR-01 acceptance.  
**Example:** Playwright already navigates `/react` and `/lit` in `dual-frontend-routes.spec.ts` [VERIFIED: file read].

### Pattern 2: Console error budget

**What:** Attach `page.on('console')` before navigation; fail test if any browser console `error` during load / scenario.  
**When to use:** Automating roadmap success criterion “no console errors on page load.”  
**Example:**

```typescript
// Source: [CITED: Playwright ConsoleMessage / Page.on('console') — Context7 /microsoft/playwright]
page.on('console', (msg) => {
  if (msg.type() === 'error') {
    throw new Error(`Console error: ${msg.text()}`);
  }
});
await page.goto('/react');
```

### Pattern 3: Strict mode verification pipeline

**What:** Run `tsc --noEmit` (project references or single project including `vite.config.ts` if in `include`) separately from `vite build`.  
**When to use:** Any claim of “zero TypeScript strict errors” in production build.  
**Evidence:** Vite does not perform type checking [CITED: https://vite.dev/guide/features.html#typescript].

### Anti-Patterns to Avoid

- **Treating `vite build` success as proof of type safety:** Build can pass with TS errors unsurfaced [CITED: vite.dev].  
- **Removing `skipLibCheck` without cause:** It skips `.d.ts` checking, not app sources; it is common in Vite templates for dependency friction [CITED: typescriptlang.org/tsconfig `#skipLibCheck`; vite.dev notes templates use `skipLibCheck`]. Dropping it may explode `node_modules` diagnostics — tune deliberately.  
- **Defining “identical outputs” as pixel-perfect:** PAR-01 is functional identity; visuals are a **separate roadmap criterion** — scope CSS work accordingly [VERIFIED: ROADMAP splits functional vs CSS].

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Browser console capture | Custom Puppeteer shim | Playwright `page.on('console')` | First-class API, matches existing test stack [CITED: Playwright docs] |
| Type checking during bundling | “Trust the bundler” | `tsc --noEmit` or checker plugin | Vite explicitly outsources type checking [CITED: vite.dev] |
| Route existence | Manual QA only | Extend `dual-frontend-routes.spec.ts` | Already guards `/react` + `/lit` shells [VERIFIED] |

**Key insight:** Parity is a **product QA** problem layered on a **typed SPA** — combine static analysis (`tsc`), runtime smoke (Playwright), and human checklist (PAR-01).

## Common Pitfalls

### Pitfall 1: `strict: true` but no `tsc` in CI

**What goes wrong:** Production bundle ships despite latent type errors.  
**Why it happens:** Vite transpiles without semantic TS checks [CITED: vite.dev].  
**How to avoid:** Add explicit `tsc --noEmit` to Phase 5 / CI validation tasks.  
**Warning signs:** `npm run build` green while IDE shows errors; `tsc` fails locally.

### Pitfall 2: Ignoring `skipLibCheck` semantics

**What goes wrong:** Assuming strict mode didn’t check “enough” of dependencies.  
**Why it happens:** Confusion between app sources vs `.d.ts` processing.  
**How to avoid:** Document that `skipLibCheck` affects declaration files, not your `.ts` sources [CITED: typescriptlang.org `#skipLibCheck`].  
**Warning signs:** Debates about removing `skipLibCheck` without measuring `node_modules` diagnostic load.

### Pitfall 3: Console noise vs real defects

**What goes wrong:** Flaky tests from benign `console.warn` or extension logs.  
**Why it happens:** Strict “no console output” policies catch DevTools noise.  
**How to avoid:** Filter on `msg.type() === 'error'` first; scope to post-load window; run headed vs CI with same args.  
**Warning signs:** Pass locally, fail in CI with font/CORS warnings upgraded to errors.

### Pitfall 4: CSS token drift between Lit and React

**What goes wrong:** Matching layout with duplicated hex values instead of shared variables.  
**Why it happens:** React CSS modules evolved independently from Lit global styles.  
**How to avoid:** Audit `var(--token)` usage vs `front-end/src/styles` / global Lit CSS; align fallbacks.  
**Warning signs:** Same semantic color with different hardcoded fallbacks in `.module.css` files [VERIFIED: grep patterns].

## Code Examples

### Playwright: fail on console error (pattern)

```typescript
// Source: [CITED: https://github.com/microsoft/playwright/blob/main/docs/src/api/class-consolemessage.md via Context7]
page.on('console', (msg) => {
  if (msg.type() === 'error') {
    console.log(`Error text: "${msg.text()}"`);
  }
});
```

### TypeScript: strict flag bundle

```json
// Source: [CITED: Context7 /microsoft/typescript — baseline strict config pattern]
{
  "compilerOptions": {
    "strict": true
  }
}
```

*(Repo already extends this via `tsconfig.base.json` [VERIFIED].)*

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single SPA entry | Vite MPA multi-html entries | Phase 1 scaffold | Dual `rollupOptions.input` for Lit+React [VERIFIED: `vite.config.ts`] |
| Manual QA only | Manual + selective Playwright | Ongoing | Hybrid fits PAR-01 wording |

**Deprecated/outdated:**

- Relying on bundler for types — never valid for strictness claims [CITED: vite.dev].

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|-----------------|
| A1 | Optional `vite-plugin-checker` is acceptable if `tsc` stays source of truth | Alternatives | Adds maintenance; may duplicate CI |
| A2 | Pixel diff tooling (e.g. Percy) is out of scope unless added later | Summary | None for PAR-01 — manual verification remains |

**Note:** Claims not listed here were verified in-session or cited above.

## Open Questions (RESOLVED)

1. **Canonical artifact directory (`05-polish-and-parity-validation/` vs `05-polish-parity/` draft folder)**  
   **Resolution:** Execution and new reports use **`.planning/phases/05-polish-and-parity-validation/`** only. Any material under `05-polish-parity/` is non-canonical draft; consolidate with `git mv` if those files are still needed [LOCKED for Phase 5 planning].

2. **Automated visual regression vs manual for CSS parity**  
   **Resolution:** **Manual** side-by-side and token/CSS audit first (Plans 05-02 / PAR-01). Optional Playwright screenshots only if CI-stable later; not required for PAR-01 [LOCKED].

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Node.js | npm, Vite, Playwright | ✓ | v22.22.2 [VERIFIED: `node --version`] | — |
| npm | scripts | ✓ | 10.9.7 [VERIFIED: `npm --version`] | — |
| `@playwright/test` | E2E / console tests | ✓ | 1.56.1 [VERIFIED: `npm ls`] | Manual browser testing |
| Vite `preview` (4173) | `playwright.config.ts` default | ✓ | via `npm run build && vite preview` in config | Set `PLAYWRIGHT_BASE_URL` to external server |
| Backend (Go TLS 8443) | Feature parity scenarios | Not probed | — | Mock/stub not applicable for “identical outputs” — real backend expected for meaningful PAR-01 [ASSUMED] |

**Missing dependencies with no fallback:**

- None identified for **static** validation (`tsc`, `vite build`, unit tests).

**Missing dependencies with fallback:**

- Full parity testing without backend: use documented dev topology (`VITE_API_ORIGIN`) [ASSUMED: matches `CLAUDE.md` / `vite.config.ts` proxy pattern].

**Step 2.6 note:** Phase is primarily validation; external backend availability is an **environment** concern for manual parity, not a new library dependency.

## Validation Architecture

> `workflow.nyquist_validation` is **true** in `.planning/config.json` [VERIFIED].

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Vitest 1.6.1 + Playwright 1.56.1 [VERIFIED: `npm ls`] |
| Config files | `front-end/vitest.config.ts`, `front-end/playwright.config.ts` [VERIFIED] |
| Quick run command | `cd front-end && npm run test -- --run` (Vitest) [VERIFIED: `package.json` `"test": "vitest"`] |
| Full suite command | `./scripts/test/unit-tests.sh --skip-golang --skip-cpp` + `./scripts/test/e2e.sh --chromium` [VERIFIED: `CLAUDE.md`] |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| PAR-01 | `/react` and `/lit` behave the same for all features | Manual + assisted | `npm run test:e2e` (partial) + checklist | Partial — `dual-frontend-routes.spec.ts` covers shells only [VERIFIED] |

### Sampling Rate

- **Per task commit:** `cd front-end && npx tsc --noEmit` + targeted Vitest [ASSUMED: project convention].  
- **Per wave merge:** `npm run build` + full Vitest run + Playwright subset.  
- **Phase gate:** PAR-01 checklist signed off; `tsc` clean; console policy met.

### Wave 0 Gaps

- [ ] **`npx tsc --noEmit` clean** — currently fails (vite.config typing) [VERIFIED: 2026-04-13 run].  
- [ ] **Console error harness** — not present as shared fixture; add helper for `/react` and `/lit` loads.  
- [ ] **Feature-level parity specs** — most e2e tests may target legacy default URL; confirm each test’s `baseURL` hits Lit shell — mapping React journeys may need new specs [ASSUMED — verify against `getBaseUrl()` usage].

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | N/A (no auth) [VERIFIED: REQUIREMENTS] |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Settings/config forms — align React with Lit validation; avoid `dangerouslySetInnerHTML` for user content [ASSUMED pattern] |
| V6 Cryptography | no | — |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| XSS via rendered HTML | Spoofing | React default escaping; avoid unsafe HTML APIs [ASSUMED: React defaults] |
| TLS MITM in dev | Tampering | Trust stores / `ignoreHTTPSErrors` only in tests (`playwright.config.ts`) [VERIFIED] — do not weaken production |

## Sources

### Primary (HIGH confidence)

- [VERIFIED] `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` — PAR-01, phase ordering, success criteria  
- [VERIFIED] `front-end/vite.config.ts`, `front-end/tsconfig.base.json`, `front-end/package.json`, `front-end/playwright.config.ts`, `front-end/vitest.config.ts`  
- [VERIFIED] `npm ls` / CLI versions in `front-end/` (2026-04-13)  
- [CITED] https://vite.dev/guide/features.html#typescript — Vite does not type-check  
- [CITED] https://www.typescriptlang.org/tsconfig/#skipLibCheck — `skipLibCheck` semantics  
- [CITED] Context7 `/microsoft/playwright` — `page.on('console')` / `ConsoleMessage`

### Secondary (MEDIUM confidence)

- [CITED] Draft Phase 5 plans under `.planning/phases/05-polish-parity/*.md` — task ideas only; may diverge from canonical phase folder naming

### Tertiary (LOW confidence)

- [ASSUMED] Backend availability and exact dev URLs for parity sessions without reading operator runbooks

## Metadata

**Confidence breakdown:**

- Standard stack: **HIGH** — pinned to installed versions and configs.  
- Architecture: **HIGH** — matches repository layout and tests.  
- Pitfalls: **MEDIUM-HIGH** — `tsc` failure reproduced; remaining risks are QA process variance.

**Research date:** 2026-04-13  
**Valid until:** ~30 days (toolchain); PAR-01 methodology stable until feature set changes post Phase 4.

---

## RESEARCH COMPLETE

**Phase:** 05 — Polish and Parity Validation  
**Confidence:** MEDIUM

### Key Findings

1. **PAR-01 is manual functional parity** across `/react` and `/lit`; automation should supplement, not redefine, the requirement [VERIFIED: REQUIREMENTS].  
2. **`vite build` ≠ typecheck** — use `tsc --noEmit` (or checker plugin) for strict-mode claims [CITED: vite.dev].  
3. **`tsc --noEmit` currently fails** in this repo (at least `vite.config.ts` vs `UserConfig`) — Phase 5 must budget fixes [VERIFIED: command].  
4. **Playwright already exposes dual-route smoke tests**; extend patterns for console errors [VERIFIED: tests + Playwright docs].  
5. **`skipLibCheck: true` is present** — understand it skips `.d.ts` checking, not application sources [CITED: TypeScript handbook].  
6. **Folder naming:** draft plans under `05-polish-parity/` vs orchestrator path `05-polish-and-parity-validation/` — align before writing reports [VERIFIED: glob].

### File Created

`.planning/phases/05-polish-and-parity-validation/05-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Standard stack | HIGH | `npm ls`, configs read |
| Architecture | HIGH | Matches codebase + e2e |
| Pitfalls | MEDIUM-HIGH | `tsc` run captured; QA subjectivity remains |

### Resolved decisions

- Canonical path: `05-polish-and-parity-validation/` (draft `05-polish-parity/` superseded for execution).
- CSS parity: manual + audit first; no mandatory pixel-diff tooling for v1.0.

### Ready for Planning

Research complete. Planner can create or align `05-xx-PLAN.md` tasks with explicit `tsc`, console, CSS audit, and PAR-01 checklist steps.
