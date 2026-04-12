# Technology Stack

**Analysis Date:** 2026-04-12

## Languages

**Primary:**
- Go 1.24.0 - Backend web server, gRPC client, WebSocket handlers, business logic
- C++17 - CUDA accelerator library, image processing kernels, CPU fallback implementations
- TypeScript 5.3.3 - Frontend web components, application logic

**Secondary:**
- CUDA - GPU kernel implementations (.cu files)
- Protocol Buffers (protobuf) - Service definitions and data contracts
- Shell (Bash) - Build scripts, deployment automation, development tools
- Python - Bazel build rules for Python dependencies

## Runtime

**Environment:**
- Go: Native execution with CGO_ENABLED=0 for production builds
- C++: Native execution via shared library (.so) loaded by Go
- CUDA: NVIDIA GPU runtime (CUDA Toolkit)
- Node.js: 20.x (for frontend build tooling only)

**Package Manager:**
- Go: Go modules (go.mod, go.sum)
- Frontend: npm (package-lock.json present)
- C++: Bazel (MODULE.bazel, MODULE.bazel.lock)
- CUDA: Bazel with rules_cuda

## Frameworks

**Core:**
- Go: Standard library HTTP server with native HTTPS
- C++: Clean Architecture with dependency injection
- Frontend: Lit 3.1.0 (Web Components)
- gRPC: ConnectRPC 1.7.0 (Go) + gRPC 1.75.0 (Go)

**Testing:**
- Go: testing stdlib + testify 1.11.1 + godog (BDD)
- Frontend: Vitest 1.2.0 + Playwright 1.40.0 (E2E)
- C++: GoogleTest 1.15.2 + GoogleMock

**Build/Dev:**
- Bazel 7.0+ - C++/CUDA build system
- Makefile - Go build orchestration
- Vite 5.0.10 - Frontend build and dev server
- Docker - Multi-stage builds for all components

## Key Dependencies

**Critical:**

**Go Backend:**
- connectrpc.com/connect v1.19.1 - gRPC-Web alternative, ConnectRPC protocol
- github.com/gorilla/websocket v1.5.3 - WebSocket server implementation
- github.com/rs/zerolog v1.34.0 - Structured logging
- github.com/spf13/viper v1.21.0 - Configuration management
- go.flipt.io/flipt-client v1.2.0 - Feature flag management
- go.opentelemetry.io/otel v1.38.0 - OpenTelemetry tracing SDK
- google.golang.org/grpc v1.75.0 - gRPC client for C++ communication
- github.com/eclipse/paho.mqtt.golang v1.5.1 - MQTT client for IoT device management
- github.com/cucumber/godog v0.14.1 - BDD testing framework

**Frontend:**
- lit 3.1.0 - Web Components base library
- @connectrpc/connect-web v1.7.0 - ConnectRPC client for browser
- @opentelemetry/sdk-trace-web v1.28.0 - Browser tracing SDK
- vite 5.0.10 - Build tool and dev server
- @playwright/test v1.40.0 - E2E testing framework

**C++ Accelerator:**
- OpenTelemetry C++ 1.18.0 - Distributed tracing
- spdlog 1.12.0 - Logging library
- nlohmann/json 3.11.2 - JSON parsing
- absl 20240116.0 - Abseil common libraries
- Lyra 1.6.1 - Command line parsing

**Infrastructure:**
- Bazel rules:
  - rules_cc 0.1.1 - C++ compilation
  - rules_cuda 0.2.5 - CUDA compilation
  - rules_proto 7.0.2 - Protocol buffer compilation
  - rules_python 0.40.0 - Python dependencies

## Configuration

**Environment:**
- YAML-based configuration per environment (dev, staging, production)
- Config files: `config/config.{env}.yaml`
- Viper for Go configuration loading
- Environment-specific .env files for frontend (development, staging, production)
- Frontend Vite proxy for gRPC-Web translation

**Key configs required:**

**Development:**
- `config/config.dev.yaml` - Go server config
- `.secrets/localhost+2.pem` and `localhost+2-key.pem` - TLS certificates
- `webserver/web/.env.development` - Frontend env vars
- Bazel workspace configuration for CUDA toolchain

**Staging:**
- `config/config.staging.yaml`
- Docker Compose services configuration
- Flipt feature flags database

**Production:**
- `config/config.production.yaml`
- NVIDIA GPU runtime availability
- MQTT broker credentials for device management
- TLS certificates for HTTPS

**Build:**
- Multi-stage Dockerfile with intermediate images from GHCR
- Bazel MODULE.bazel for C++ dependencies
- Go Makefile for Go builds
- Vite configuration for frontend
- Proto generation via buf CLI

## Platform Requirements

**Development:**
- Go 1.24+
- Node.js 20+
- CUDA Toolkit 11.x+ (for GPU development)
- Bazel 7.0+
- Docker 20.x+ (with BuildKit)
- NVIDIA Container Toolkit (for GPU containers)
- OpenSSL for local TLS certificates

**Production:**
- Linux with NVIDIA GPU driver
- CUDA runtime libraries
- Docker runtime with NVIDIA support
- 8GB+ RAM recommended
- 4GB+ GPU VRAM for CUDA processing
- TLS certificates for HTTPS

---

*Stack analysis: 2026-04-12*
