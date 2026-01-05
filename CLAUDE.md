# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUDA Learning Platform - Real-time image/video processing via CUDA GPU kernels. Go web server communicates with C++/CUDA accelerator library via gRPC for distributed processing.

## Build & Development Commands

### Development Environment
```bash
./scripts/dev/start.sh --build  # First time: build C++/Go + start dev server
./scripts/dev/start.sh          # Subsequent runs (hot reload enabled)
./scripts/dev/stop.sh           # Stop all services
```
Access: https://localhost:8443

### Building Components
```bash
# C++ (Bazel)
bazel build //cpp_accelerator/ports/grpc:image_processor_grpc_server
bazel build //cpp_accelerator/...

# Go (from webserver/)
cd webserver && make build

# Frontend (from webserver/web/)
cd webserver/web && npm install && npm run dev
```

### Testing
```bash
# All coverage tests
./scripts/test/coverage.sh

# Unit tests
./scripts/test/unit-tests.sh
./scripts/test/unit-tests.sh --skip-golang   # Frontend only
./scripts/test/unit-tests.sh --skip-frontend # Go only

# Individual test suites
go test -race ./webserver/pkg/...                    # Go tests
cd webserver/web && npm run test                     # Frontend (Vitest)
bazel test //cpp_accelerator/...                     # C++ tests

# Specific C++ test
bazel test //cpp_accelerator/core:logger_test

# BDD acceptance tests (requires services running)
go test ./integration/tests/acceptance -run TestFeatures -v

# E2E tests
./scripts/test/e2e.sh --chromium   # Fast: Chromium only
./scripts/test/e2e.sh              # All browsers
```

### Linting
```bash
./scripts/test/linters.sh          # All linters
./scripts/test/linters.sh --fix    # Auto-fix
```

## Architecture

### Code Structure
```
cpp_accelerator/
  application/         # Use cases, FilterPipeline, BufferPool
  domain/interfaces/   # IFilter, ImageBuffer, IImageProcessor
  infrastructure/cuda/ # CUDA kernel implementations
  infrastructure/cpu/  # CPU fallback implementations
  ports/grpc/         # gRPC service (primary integration)
  ports/shared_lib/   # Shared library exports
  core/               # Logger, Telemetry, Result type

webserver/
  cmd/server/         # main.go entry point
  pkg/app/            # Application bootstrap
  pkg/application/    # Use cases
  pkg/domain/         # Domain logic
  pkg/infrastructure/ # Repositories, gRPC client
  pkg/interfaces/     # HTTP/WebSocket handlers
  web/                # Frontend (Lit Web Components + Vite)
```

### Processing Flow
Go Server → gRPC Client → gRPC Server (C++) → ProcessorEngine → FilterPipeline → CUDA/CPU Filters

### Key Patterns
- Clean Architecture with dependency injection
- FilterPipeline orchestrates composable filter chains
- ProcessorEngine coordinates between ports and pipeline
- gRPC streaming for video processing (StreamProcessVideo)

## Testing Patterns

### Go Tests
- Use AAA comments (Arrange/Act/Assert)
- Use `testify/mock` (embed `mock.Mock`)
- Use `sut` for system under test
- Table-driven tests with `assertResult` function
- Naming: `Success_`, `Error_`, `Edge_` prefix
- Test data builders: `makeXXX()`

### TypeScript Tests (Vitest)
- Use AAA comments
- Use `vi.fn()` and `vi.mock()`
- Use `sut` for system under test
- Table-driven tests for multiple cases
- Naming: `Success_`, `Error_`, `Edge_` prefix
- Test data builders: `makeXXX()`

### C++ Tests (GoogleTest)
- Bazel test targets: `//cpp_accelerator/path:target_test`
- Equivalence tests verify CPU/CUDA produce identical results

## Configuration

- Development: `config/config.dev.yaml`
- Staging: `config/config.staging.yaml`
- Production: `config/config.production.yaml`

Proto generation: `./scripts/build/protos.sh`

## Tech Stack

- **Backend**: Go with native HTTPS, WebSocket support
- **Processing**: C++/CUDA via gRPC service (ConnectRPC)
- **Build**: Bazel for C++/CUDA, Makefile for Go
- **Frontend**: Lit Web Components + TypeScript with Vite
- **Observability**: OpenTelemetry, Jaeger tracing, Grafana dashboards, Loki logs

## Git Hooks

```bash
./scripts/hooks/install.sh   # Install pre-commit/pre-push hooks
git commit --no-verify       # Skip hooks when needed
```

Pre-commit: unit tests + linters
Pre-push: full validation with all browsers
