# Codebase Structure

**Analysis Date:** 2025-04-12

## Directory Layout

```
cuda-learning/
├── cpp_accelerator/          # C++ image processing engine with CUDA
│   ├── application/          # Use cases, FilterPipeline, BufferPool
│   ├── core/                 # Logger, Telemetry, Result type
│   ├── domain/               # Domain interfaces and models
│   ├── infrastructure/       # CUDA/CPU implementations, image I/O
│   ├── ports/                # gRPC server, shared library exports
│   └── cmd/                  # C++ executables
├── webserver/                # Go web server and frontend
│   ├── cmd/server/           # main.go entry point
│   ├── pkg/
│   │   ├── app/              # Application bootstrap
│   │   ├── application/      # Use cases
│   │   ├── domain/           # Domain interfaces
│   │   ├── infrastructure/   # Repositories, gRPC client
│   │   ├── interfaces/       # HTTP/WebSocket handlers
│   │   ├── container/        # DI container
│   │   └── config/           # Configuration loading
│   ├── web/                  # Frontend (Lit Web Components)
│   │   └── src/
│   │       ├── components/   # UI components
│   │       ├── application/  # Frontend DI container
│   │       ├── infrastructure/ # Transport, logging
│   │       └── services/     # Business logic services
│   └── config/               # Server configuration
├── proto/                    # Protocol Buffer definitions
│   ├── image_processor_service.proto
│   ├── config_service.proto
│   ├── file_service.proto
│   └── remote_management_service.proto
├── config/                   # YAML configuration files
│   ├── config.yaml
│   ├── config.dev.yaml
│   ├── config.staging.yaml
│   └── config.production.yaml
├── scripts/                  # Build, test, deployment scripts
│   ├── build/
│   ├── dev/
│   ├── test/
│   ├── linters/
│   └── deployment/
├── integration/              # BDD acceptance tests
│   └── tests/acceptance/     # Godog features and steps
├── docs/                     # Documentation
├── third_party/              # Third-party dependencies for Bazel
├── bazel/                    # Bazel configuration
└── data/                     # Runtime data (videos, previews)
```

## Directory Purposes

**cpp_accelerator/:**
- Purpose: C++ image processing engine with CUDA acceleration
- Contains: Core image processing logic, GPU kernels, gRPC server
- Key files:
  - `ports/grpc/server_main.cpp`: gRPC server entry point
  - `ports/shared_lib/processor_engine.h`: Core processing engine
  - `application/pipeline/filter_pipeline.h`: Filter orchestration
  - `infrastructure/cuda/`: CUDA kernel implementations
  - `infrastructure/cpu/`: CPU fallback implementations

**webserver/:**
- Purpose: Go web server orchestrating processing and serving UI
- Contains: HTTP server, WebSocket handlers, gRPC client, frontend assets
- Key files:
  - `cmd/server/main.go`: Server entry point
  - `pkg/app/app.go`: Application bootstrap and route setup
  - `pkg/container/container.go`: Dependency injection
  - `pkg/infrastructure/processor/grpc_client.go`: C++ client
  - `web/src/main.ts`: Frontend entry point

**proto/:**
- Purpose: Protocol Buffer definitions for cross-language contracts
- Contains: Service definitions, message types
- Key files:
  - `image_processor_service.proto`: Core image processing service
  - `config_service.proto`: Configuration and feature flag service
  - `file_service.proto`: File upload/listing service
  - `remote_management_service.proto`: Remote management capabilities
  - `gen/`: Generated code (Go, TypeScript, C++)

**config/:**
- Purpose: Environment-specific configuration files
- Contains: YAML configs for dev/staging/production
- Key files:
  - `config.yaml`: Default configuration
  - `config.dev.yaml`: Development overrides
  - `config.staging.yaml`: Staging environment
  - `config.production.yaml`: Production settings

**scripts/:**
- Purpose: Automation for build, test, deployment, and development
- Contains: Shell scripts for all workflows
- Key files:
  - `dev/start.sh`: Start development environment
  - `dev/stop.sh`: Stop development services
  - `test/coverage.sh`: Run all tests with coverage
  - `test/unit-tests.sh`: Run unit tests
  - `test/linters.sh`: Run linters
  - `build/protos.sh`: Generate protobuf code

**integration/:
- Purpose: BDD acceptance tests validating end-to-end functionality
- Contains: Godog features, step definitions, test data
- Key files:
  - `tests/acceptance/features/`: Feature files
  - `tests/acceptance/steps/`: Step definitions

## Key File Locations

**Entry Points:**
- `webserver/cmd/server/main.go`: Go HTTP server
- `cpp_accelerator/ports/grpc/server_main.cpp`: C++ gRPC server
- `webserver/web/src/main.ts`: Frontend application
- `cpp_accelerator/cmd/hello_cuda/`: Simple CUDA test program

**Configuration:**
- `webserver/pkg/config/config.go`: Configuration loading logic
- `config/config.yaml`: Default configuration values
- `cpp_accelerator/infrastructure/config/config_manager.h`: C++ config management

**Core Logic:**
- `cpp_accelerator/application/pipeline/filter_pipeline.h`: Filter orchestration
- `cpp_accelerator/ports/shared_lib/processor_engine.h`: Processing engine facade
- `webserver/pkg/application/process_image_use_case.go`: Image processing use case
- `webserver/pkg/infrastructure/processor/grpc_client.go`: gRPC client

**Testing:**
- `cpp_accelerator/**/*_test.cpp`: C++ unit tests (GoogleTest)
- `webserver/pkg/**/*_test.go`: Go unit tests
- `webserver/web/src/**/*.test.ts`: TypeScript unit tests (Vitest)
- `integration/tests/acceptance/`: BDD tests (Godog)

## Naming Conventions

**Files:**
- C++ headers: `snake_case.h` (e.g., `image_processor_service_impl.h`)
- C++ source: `snake_case.cpp` (e.g., `image_processor_service_impl.cpp`)
- Go source: `snake_case.go` (e.g., `process_image_use_case.go`)
- TypeScript: `kebab-case.ts` or `camelCase.ts` (e.g., `frame-transport-service.ts`)
- Protobuf: `snake_case.proto` (e.g., `image_processor_service.proto`)
- Test files: Append `_test.go`, `_test.cpp`, `.test.ts`

**Directories:**
- Lowercase with underscores: `image_processor_service`, `grpc_client`
- Exception: `webserver/web/src` uses some hyphens for consistency with npm

**Go Packages:**
- Lowercase single words: `app`, `config`, `domain`
- Multi-word: `infrastructure/processor`, `interfaces/connectrpc`

**C++ Namespaces:**
- Nested by directory: `jrb::application::pipeline`, `jrb::domain::interfaces`
- Class names: `PascalCase` (e.g., `FilterPipeline`, `ProcessorEngine`)

**TypeScript/JavaScript:**
- Classes: `PascalCase` (e.g., `FrameTransportService`)
- Functions/variables: `camelCase` (e.g., `processImage`, `grpcClient`)
- Interfaces: `IPascalCase` (e.g., `IConfigService`, `ILogger`)
- Custom elements: `kebab-case` (e.g., `camera-preview`, `filter-panel`)

**Protocol Buffers:**
- Messages: `PascalCase` (e.g., `ProcessImageRequest`, `FilterType`)
- Fields: `snake_case` (e.g., `image_data`, `cuda_device_id`)
- Services: `PascalCase` (e.g., `ImageProcessorService`)

## Where to Add New Code

**New Feature (Backend - Go):**
- Primary code: `webserver/pkg/application/[feature]_use_case.go`
- Domain interface: `webserver/pkg/domain/[feature]_repository.go`
- Implementation: `webserver/pkg/infrastructure/[type]/[feature]_*.go`
- Handler: `webserver/pkg/interfaces/connectrpc/[feature]_handler.go`
- Tests: `webserver/pkg/application/[feature]_use_case_test.go`

**New Feature (Backend - C++):**
- Domain interface: `cpp_accelerator/domain/interfaces/[feature].h`
- Implementation: `cpp_accelerator/infrastructure/[type]/[feature].h` and `.cpp`
- Integration: `cpp_accelerator/application/commands/[feature]_command.h`
- Port adapter: `cpp_accelerator/ports/grpc/[feature]_service_impl.h` and `.cpp`
- Tests: `cpp_accelerator/[path]/[feature]_test.cpp`

**New Feature (Frontend):**
- Component: `webserver/web/src/components/[category]/[name].ts`
- Service: `webserver/web/src/services/[name]-service.ts`
- Domain model: `webserver/web/src/domain/[name].ts`
- Infrastructure: `webserver/web/src/infrastructure/[category]/[name].ts`
- Tests: `webserver/web/src/components/[category]/[name].test.ts`

**New Filter (Image Processing):**
- Interface: `cpp_accelerator/domain/interfaces/filters/i_[filter]_filter.h`
- CUDA implementation: `cpp_accelerator/infrastructure/cuda/[filter]_kernel.h` and `.cu`
- CPU implementation: `cpp_accelerator/infrastructure/cpu/[filter]_filter.h` and `.cpp`
- Tests: `cpp_accelerator/infrastructure/cuda/[filter]_kernel_test.cpp`

**New Proto Service:**
- Definition: `proto/[service]_service.proto`
- Generate code: `./scripts/build/protos.sh`
- Go handler: `webserver/pkg/interfaces/connectrpc/[service]_handler.go`
- C++ service: `cpp_accelerator/ports/grpc/[service]_service_impl.cpp`
- TypeScript client: Generated in `webserver/web/src/gen/`

**Utilities:**
- Shared helpers: `cpp_accelerator/core/` (C++) or `webserver/pkg/infrastructure/[category]/` (Go)
- Frontend utilities: `webserver/web/src/infrastructure/`

## Special Directories

**third_party/:**
- Purpose: Third-party dependencies for Bazel build
- Generated: No
- Committed: Yes
- Contains: STB image library, Lyra CLI parser

**proto/gen/:**
- Purpose: Generated Protocol Buffer code
- Generated: Yes (by `./scripts/build/protos.sh`)
- Committed: Yes
- Contains: Go, TypeScript, C++ generated code

**data/:**
- Purpose: Runtime data files (videos, previews, uploaded images)
- Generated: Partially (previews generated on upload)
- Committed: Partially (sample videos yes, user uploads no)
- Contains: `videos/`, `video_previews/`, `static_images/`

**.planning/:**
- Purpose: Planning documents for development
- Generated: Yes (by GSD commands)
- Committed: Yes
- Contains: `codebase/` (analysis documents), `research/`, `phases/`

**coverage/:**
- Purpose: Test coverage reports
- Generated: Yes (by `./scripts/test/coverage.sh`)
- Committed: No
- Contains: HTML coverage reports for Go, TypeScript, C++

**.secrets/:**
- Purpose: Local development secrets (TLS certificates, keys)
- Generated: Yes (by `./scripts/secrets/generate.sh`)
- Committed: No (gitignored)
- Contains: `keys/`, `certs/`, `cloudflare/`

**webserver/web/dist/:**
- Purpose: Production frontend build output
- Generated: Yes (by `npm run build`)
- Committed: No
- Contains: Minified JS, CSS, assets

**webserver/web/node_modules/:**
- Purpose: NPM dependencies
- Generated: Yes (by `npm install`)
- Committed: No
- Contains: Frontend packages

**bazel-out/:**
- Purpose: Bazel build output
- Generated: Yes (by `bazel build`)
- Committed: No
- Contains: Compiled C++ binaries and objects

---

*Structure analysis: 2025-04-12*
