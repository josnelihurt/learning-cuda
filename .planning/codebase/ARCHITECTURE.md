# Architecture

**Analysis Date:** 2025-04-12

## Pattern Overview

**Overall:** Clean Architecture with Hexagonal Ports and Adapters

**Key Characteristics:**
- Domain-driven design with clear separation of concerns
- C++ accelerator service communicates via gRPC (ConnectRPC protocol)
- Go web server orchestrates processing and serves UI
- TypeScript frontend uses Lit Web Components
- Dependency injection throughout all layers
- Protocol Buffers define cross-language contracts

## Layers

**Domain Layer (C++):**
- Purpose: Core business logic and interfaces
- Location: `cpp_accelerator/domain/`
- Contains: Interface definitions (`IImageProcessor`, `IFilter`, `ImageBuffer`), algorithm abstractions
- Depends on: Nothing (core layer)
- Used by: Application layer, Infrastructure implementations

**Application Layer (C++):**
- Purpose: Use case orchestration and business workflows
- Location: `cpp_accelerator/application/`
- Contains: `FilterPipeline` (orchestrates filter chains), `BufferPool` (memory management), command factory
- Depends on: Domain interfaces
- Used by: Ports (gRPC service, shared library)

**Infrastructure Layer (C++):**
- Purpose: External concerns and concrete implementations
- Location: `cpp_accelerator/infrastructure/`
- Contains:
  - `cuda/`: CUDA kernel implementations
  - `cpu/`: CPU fallback implementations
  - `image/`: Image loading/writing (STB)
  - `config/`: Configuration management
- Depends on: Domain interfaces
- Used by: Application layer

**Ports Layer (C++):**
- Purpose: Integration adapters for external communication
- Location: `cpp_accelerator/ports/`
- Contains:
  - `grpc/`: gRPC server (ConnectRPC), `ImageProcessorServiceImpl`, `ProcessorEngineAdapter`
  - `shared_lib/`: Shared library exports for direct linking
- Depends on: Application layer, domain interfaces
- Used by: Go web server (via gRPC), external clients

**Domain Layer (Go):**
- Purpose: Core Go interfaces and domain models
- Location: `webserver/pkg/domain/`
- Contains: `ImageProcessor`, `VideoRepository`, `FeatureFlagRepository`, domain entities
- Depends on: Nothing
- Used by: Application layer, Infrastructure implementations

**Application Layer (Go):**
- Purpose: Use cases coordinating domain operations
- Location: `webserver/pkg/application/`
- Contains: `ProcessImageUseCase`, `ProcessorCapabilitiesUseCase`, `ListVideosUseCase`, etc.
- Depends on: Domain interfaces, infrastructure repositories
- Used by: HTTP/WebSocket handlers, ConnectRPC services

**Infrastructure Layer (Go):**
- Purpose: External service integrations and technical concerns
- Location: `webserver/pkg/infrastructure/`
- Contains:
  - `processor/`: gRPC client to C++ accelerator
  - `filesystem/`: File system repositories
  - `mqtt/`: MQTT device monitoring
  - `logger/`: Structured logging (OTLP support)
  - `video/`: Video management
  - `config/`: Configuration loading
- Depends on: Domain interfaces
- Used by: Application layer

**Interface Layer (Go):**
- Purpose: HTTP/WebSocket/ConnectRPC handlers
- Location: `webserver/pkg/interfaces/`
- Contains:
  - `connectrpc/`: ConnectRPC service implementations
  - `http/`: REST handlers (health, trace/logs proxy)
  - `websocket/`: WebSocket handlers (WebRTC signaling)
  - `statichttp/`: Static file serving, SPA routing
- Depends on: Application layer use cases
- Used by: HTTP server

**Frontend Layer (TypeScript):**
- Purpose: Client-side UI and business logic
- Location: `webserver/web/src/`
- Contains:
  - `components/`: Lit Web Components (custom elements)
  - `application/`: Frontend use cases and DI container
  - `infrastructure/`: External services (transport, logging)
  - `services/`: Business logic services
  - `domain/`: Frontend domain models
- Depends on: Generated protobuf clients
- Used by: Browser

## Data Flow

**Image Processing Flow:**

1. Browser sends request via ConnectRPC to Go server (`/cuda_learning.ImageProcessorService/ProcessImage`)
2. Go `ProcessImageUseCase` receives request, creates `ProcessingOptions`
3. `GRPCProcessor` adapts request to protobuf `ProcessImageRequest`
4. gRPC client calls C++ service at `cpp_accelerator/ports/grpc/image_processor_service_impl.cpp`
5. `ImageProcessorServiceImpl` delegates to `ProcessorEngineAdapter`
6. Adapter creates `ProcessImageCommand` via `CommandFactory`
7. `ProcessorEngine` validates and applies filters
8. `FilterPipeline` orchestrates filter execution:
   - Acquires intermediate buffers from `BufferPool`
   - Applies each filter (CUDA or CPU implementation)
   - Releases intermediate buffers
9. Result returned through gRPC to Go server
10. Go server returns protobuf response to browser
11. Browser displays processed image

**Video Streaming Flow (WebRTC):**

1. Browser establishes WebRTC connection via WebSocket signaling
2. Browser sends `StartVideoPlaybackRequest` with filters
3. Go server spawns video processing session
4. For each video frame:
   - Frame decoded by Go server
   - Sent via gRPC streaming to C++ service (`StreamProcessVideo`)
   - C++ processes frame with filter pipeline
   - Processed frame returned via stream
   - Go server forwards to browser via WebRTC data channel
5. WebSocket used for signaling (SDP offer/answer, ICE candidates)

**Configuration Flow:**

1. Application starts → `container.New()` loads `config/config.yaml`
2. Go server initializes dependencies (gRPC client, repositories, use cases)
3. Use cases injected into `app.App` via functional options
4. HTTP handlers registered with `http.ServeMux`
5. Server starts HTTP/HTTPS listeners

**State Management:**

- Go: Stateless server design, state in repositories (filesystem)
- C++: Stateless processing engine, GPU state managed per request
- Frontend: Service-oriented state management via DI container

## Key Abstractions

**FilterPipeline:**
- Purpose: Orchestrates composable filter chains with automatic buffer management
- Examples: `cpp_accelerator/application/pipeline/filter_pipeline.h`
- Pattern: Pipeline pattern with buffer pooling for memory efficiency

**ProcessorEngine:**
- Purpose: Core processing engine coordinating between ports and pipeline
- Examples: `cpp_accelerator/ports/shared_lib/processor_engine.h`
- Pattern: Facade pattern exposing simplified API

**Use Cases:**
- Purpose: Encapsulate business logic workflows
- Examples: `webserver/pkg/application/process_image_use_case.go`
- Pattern: Command pattern with dependency injection

**Repositories:**
- Purpose: Abstract data access
- Examples: `webserver/pkg/domain/video_repository.go`, `webserver/pkg/infrastructure/video/file_video_repository.go`
- Pattern: Repository pattern with interface abstraction

**ImageProcessor Interface:**
- Purpose: Unified image processing abstraction across languages
- Examples:
  - Go: `webserver/pkg/domain/processor.go`
  - C++: `cpp_accelerator/domain/interfaces/processors/i_image_processor.h`
- Pattern: Bridge pattern connecting Go and C++ implementations

## Entry Points

**Go Server:**
- Location: `webserver/cmd/server/main.go`
- Triggers: HTTP server startup on configured ports
- Responsibilities:
  - Load configuration
  - Initialize DI container
  - Setup telemetry (OpenTelemetry)
  - Create use cases with dependencies
  - Start HTTP/HTTPS listeners
  - Handle graceful shutdown

**C++ gRPC Server:**
- Location: `cpp_accelerator/ports/grpc/server_main.cpp`
- Triggers: Executable starts gRPC server
- Responsibilities:
  - Initialize CUDA device
  - Create ProcessorEngine
  - Register gRPC services (ImageProcessor, WebRTC Signaling)
  - Start gRPC listener
  - Handle shutdown signals

**Frontend Application:**
- Location: `webserver/web/src/main.ts`
- Triggers: Browser DOMContentLoaded event
- Responsibilities:
  - Initialize DI container and services
  - Register Lit Web Components
  - Setup health monitoring
  - Configure telemetry and logging

**Bazel Build Targets:**
- Location: Various `BUILD` files throughout `cpp_accelerator/`
- Triggers: `bazel build` command
- Responsibilities: Define C++ compilation rules, dependencies, and test targets

## Error Handling

**Strategy:** Result types and explicit error propagation

**Patterns:**

**C++:**
- Custom `Result<T, E>` type in `cpp_accelerator/core/result.h`
- Boolean return values with output parameters for simple operations
- Protobuf status codes in gRPC responses

**Go:**
- Standard error interface return values
- Wrapped errors with context using `fmt.Errorf`
- HTTP status codes mapped from domain errors in handlers
- gRPC status codes propagated from C++ service

**TypeScript:**
- Try-catch for async operations
- Service methods return typed responses or throw errors
- UI components display error messages via toast notifications

**Cross-cutting error handling:**
- Structured logging at all layers (JSON format)
- OpenTelemetry trace context propagation across gRPC boundary
- Health check endpoint for load balancer probes
- Graceful degradation when optional services fail (e.g., feature flags, telemetry)

## Cross-Cutting Concerns

**Logging:**
- C++: spdlog with custom OTLP sink (`cpp_accelerator/core/logger.h`)
- Go: zerolog with structured fields (`webserver/pkg/infrastructure/logger/`)
- TypeScript: Custom logger service with environment-aware levels
- All layers support correlation IDs via trace context

**Validation:**
- Protocol Buffer validation at gRPC boundaries
- Domain-level validation in use cases
- Image buffer validation (dimensions, channel counts) before processing
- Configuration validation on startup

**Authentication:**
- Not currently implemented (development/educational system)
- TLS/HTTPS for transport encryption
- Future: JWT or API key authentication planned

**Observability:**
- OpenTelemetry tracing across Go, C++, and TypeScript
- Distributed trace context propagation via protobuf
- Metrics exported to OTEL collector
- Logs shipped to Loki via OTLP
- Grafana dashboards for visualization

**Resource Management:**
- C++: Buffer pool for intermediate image buffers (reduces allocations)
- Go: HTTP client connection pooling
- TypeScript: WebRTC session lifecycle management
- Graceful shutdown handling in all services

---

*Architecture analysis: 2025-04-12*
