# Phase 1: Scaffold and Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-13
**Phase:** 01-scaffold-and-infrastructure
**Mode:** discuss
**Areas discussed:** Context update for production architecture

## Context Update Session (2026-04-13)

### Original Issue
User feedback: "I have extracted the front-end from initial location to a separate folder in root of project, decoupling go backend from frontend, and viceversa. I want to develop front-end separated from backend. many files were moved/updated. And plan initially contemplated that go expects some templates that are not longer required. So please go over milestone and close gaps related to this elements, and clean all places were we need to include this as part of requirements, now I also need to add more tasks to migration from lit to react"

### Analysis Performed
Investigated the new frontend separation architecture:

1. **Frontend structure**: `front-end/` directory with separate `package.json`, `vite.config.ts`, build configs
2. **Production serving**: Nginx static file server (not Go templates)
3. **Edge routing**: Traefik routes to Nginx, which proxies backend API to Go
4. **Vite build**: Produces `index.html` (Lit) and `react.html` (React) with HTML-based manifest keys
5. **Go role**: Only handles API/gRPC/WebSocket — NO frontend routing in production

### Corrections Made

| Original Assumption | Correction | Evidence |
|-------------------|------------|----------|
| D-08, D-09, D-10: Go routing in production (template parsing, route-aware handlers) | Nginx serves static files directly, Go only handles backend | `front-end/nginx.conf`, `docker-compose.yml` |
| D-17: Go production handler validation for asset manifest | Nginx serves static HTML files — no Go involvement | `front-end/Dockerfile` multi-stage build |
| Production asset manifest lookup via Go code | Nginx serves static files, Vite manifest uses HTML paths | `front-end/dist/.vite/manifest.json` |

### New Decisions Added

| Decision | Rationale |
|----------|-----------|
| D-22: Frontend as separate service | Frontend is now `front-end/` with own build, deployment, runtime |
| D-23: Traefik → Nginx → Go service mesh | Production traffic flows through edge → static server → backend |
| D-24: Go no longer serves frontend | Go backend only handles API/gRPC/WebSocket requests |
| D-25: Lit code coexists during migration | Lit remains in `front-end/src/` alongside React until parity achieved |

### User's Choice
"ok" — Proceed with context update

## Context Update Summary

Updated CONTEXT.md with corrected production architecture assumptions. Removed outdated Go routing assumptions and added proper Nginx/Traefik service mesh architecture.

## Updated Canonical References

Added:
- `front-end/nginx.conf` — Production static file server config
- `front-end/Dockerfile` — Multi-stage build with Nginx runtime
- `docker-compose.yml` — Full service orchestration
- `traefik-config.yml` — Edge router TLS and routing

## Updated Integration Points

Added:
- Nginx static file serving (production)
- Traefik edge routing
- Go dev proxy route awareness
- Frontend separation as service

## Deferred Ideas

None — discussion stayed within phase scope

