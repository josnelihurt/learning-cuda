# CUDA Learning Platform - React Frontend Migration

## What This Is

A learning-focused project to rebuild the frontend using React while maintaining the existing CUDA image/video processing platform. The current Lit-based frontend will serve as a reference implementation, with both frontends coexisting during development via separate routes (`/lit` and `/react`). This is a brownfield migration: the Go backend and C++/CUDA accelerator remain unchanged.

## Core Value

**Practice React best practices while delivering a production-ready frontend** — Learn modern React patterns (custom hooks, context, component design) through real-world implementation, achieving full feature parity with the existing Lit frontend.

## Requirements

### Validated

*Existing functionality from the current Lit frontend:*

- ✓ **Real-time image processing** — User can upload images and apply filters via CUDA-accelerated backend (existing)
- ✓ **Video streaming with WebRTC** — User can stream video with real-time filter application (existing)
- ✓ **Filter management** — User can select and configure from available filters (grayscale, blur, edge detection, etc.) (existing)
- ✓ **Frame-by-frame preview** — User can see processed frames in real-time (existing)
- ✓ **File upload/management** — User can upload images and videos, list available files (existing)
- ✓ **Configuration UI** — User can view and modify system settings via frontend (existing)
- ✓ **gRPC communication** — Frontend communicates with Go backend via ConnectRPC (existing)
- ✓ **Responsive design** — UI works across desktop browsers (existing)
- ✓ **Error handling** — User receives clear error messages via toast notifications (existing)
- ✓ **Health monitoring** — Frontend monitors backend health status (existing)

### Active

*New React frontend to build:*

- [ ] **React frontend with full feature parity** — Replicate all existing Lit frontend functionality in React
- [ ] **Custom hooks for logic extraction** — Extract business logic into reusable custom hooks (useImageProcessing, useVideoStream, useFilters)
- [ ] **Context-based state management** — Avoid prop drilling using React Context or Zustand for global state
- [ ] **Clean component architecture** — Separate presentational from container components, establish clear component boundaries
- [ ] **TypeScript throughout** — Maintain strict type safety in all React components and hooks
- [ ] **Vite build system** — Keep using Vite for development and production builds
- [ ] **Test migration** — Migrate existing Vitest tests to React Testing Library
- [ ] **CSS reuse** — Reuse existing CSS with minimal adaptation for React components
- [ ] **Separate development routes** — `/lit` serves existing frontend, `/react` serves new React frontend
- [ ] **gRPC client integration** — Reuse existing protobuf definitions and ConnectRPC client
- [ ] **Production-ready quality** — Performance parity with Lit, proper error boundaries, loading states
- [ ] **Code quality standards** — ESLint, Prettier, TypeScript strict mode, component documentation

### Out of Scope

- **Backend modifications** — Go server and C++ accelerator remain unchanged (reuse existing APIs)
- **New features** — Only feature parity, no new capabilities beyond what Lit frontend provides
- **Mobile apps** — Web-only, responsive design for desktop browsers
- **Authentication/authorization** — Not implemented in current system, out of scope for migration
- **Multi-language support** — English only, same as existing system

## Context

**Existing System:**
- Clean Architecture with Go backend, C++/CUDA accelerator, Lit Web Components frontend
- gRPC (ConnectRPC protocol) for all frontend-backend communication
- Protocol Buffers define cross-language contracts
- Real-time video processing via WebRTC data channels
- Vite for frontend build, Vitest for testing, Playwright for E2E
- OpenTelemetry observability across all layers

**Migration Strategy:**
- Use Lit frontend as living reference — parallel development, not a rewrite
- Extract component logic patterns from Lit and adapt to React idioms
- Reuse existing CSS, protobuf clients, and build infrastructure
- Maintain API compatibility — zero backend changes required
- Gradual rollout via routing, eventual cutover to React-only

**Learning Goals:**
- Practice modern React patterns in production codebase
- Understand differences between Web Components and React paradigms
- Learn component design and state management at scale
- Experience migrating a real application, not a tutorial

## Constraints

- **Technology Stack**: Must use React, TypeScript, Vite — no framework changes
- **Backend Compatibility**: Zero changes to Go/C++ backend, reuse existing gRPC APIs
- **Feature Parity**: Must match all existing functionality, no regressions
- **Performance**: React frontend must match or exceed Lit performance metrics
- **Code Quality**: Follow React best practices from day one (clean architecture, separation of concerns)
- **Testing**: All existing tests must migrate to React Testing Library
- **Development Experience**: Both frontends must run simultaneously during development
- **Timeline**: Learning-focused, no deadline — prioritize understanding over speed

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Separate routes (/lit, /react) | Allows parallel development and direct comparison during migration | ✓ Good |
| Reuse CSS from Lit frontend | Faster migration, consistent look-and-feel, focus on component logic | — Pending |
| Custom hooks over prop drilling | Practice React best practices, maintainable state management | — Pending |
| Context for global state | Avoid prop drilling, establish scalable state pattern | — Pending |
| Zero backend changes | Reduces risk, allows frontend-focused learning, easier rollback | ✓ Good |
| Migrate to React Testing Library | Standard React testing approach, better component testing practices | — Pending |

## Current Milestone: v1.0 React Frontend Migration

**Goal:** Build a production-ready React frontend with full feature parity to the existing Lit frontend, serving as a hands-on React learning project.

**Target features:**
- React + Vite + TypeScript scaffold with dual routing (/lit and /react)
- Clean component architecture (presentational vs container, no prop drilling)
- Custom hooks for business logic (useImageProcessing, useVideoStream, useFilters)
- Context-based global state management
- Reuse existing CSS and ConnectRPC/protobuf gRPC client
- Full feature parity: image processing, video streaming, filter management, file upload, settings, health monitoring
- React Testing Library — migrate existing Vitest frontend tests
- End state: React-only frontend after parity confirmed

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-12 after milestone v1.0 started*
