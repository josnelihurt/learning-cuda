# CUDA Image Processor - Web Server

GPU-accelerated image processing web application using CUDA, Go, C++, and Protocol Buffers.

## Development Mode

For local development with hot reload:

### Quick Start

From project root:
```bash
./scripts/dev/start.sh --build  # First time or after code changes
./scripts/dev/start.sh           # Subsequent runs
```

This will:
1. Build the server and frontend
2. Start services (Goff feature flags, Jaeger, etc.)
3. Enable hot reload for frontend (Vite dev server)
4. Start the Go server with hot reload support

**Access:** https://localhost:8443

### Manual Build

Build the Go server from `src/go_api/`:
```bash
cd src/go_api
make build
```

Or from project root:
```bash
cd src/go_api && make build
```

Run the server:
```bash
cd src/go_api
make run
```

Or with config file (from project root):
```bash
./bin/server -config=config/config.yaml
```

### Development Mode

Run server with hot reload:
```bash
cd src/go_api
make dev
```

## Production Mode

For production (embedded files, single binary):

```bash
cd src/go_api
make build
./bin/server -config=../../config/config.production.yaml
```

The frontend is built with Vite and embedded as static assets. Templates and static files are served from the binary.

## Architecture

The web server implements Clean Architecture with clear separation of concerns across four main layers: Interfaces, Application, Domain, and Infrastructure.

### Component Overview

The web server follows Clean Architecture principles with clear separation between interfaces, application logic, domain models, and infrastructure. It integrates with the C++ CUDA accelerator library via gRPC.

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
        API[API Clients]
    end
    
    subgraph "Go Interfaces Layer"
        ConnectRPC[Connect-RPC Handlers]
        WebRTC[WebRTC Handlers]
        AuxHTTP[Aux HTTP: health, logs, trace, data proxy]
        Vanguard[Vanguard Transcoder]
    end
    
    subgraph "Go Application Layer"
        ProcessImageUC[ProcessImageUseCase]
        CapabilitiesUC[ProcessorCapabilitiesUseCase]
        ConfigUC[Config Use Cases]
        FileUC[File Use Cases]
    end
    
    subgraph "Go Domain Layer"
        ImageProcessor[ImageProcessor Interface]
        Image[Image Domain Model]
        Video[Video Domain Model]
        FeatureFlag[FeatureFlag Domain]
    end
    
    subgraph "Go Infrastructure Layer"
        GRPCProcessor[GRPCProcessor]
        FeatureFlags[Goff Integration]
        FileSystem[File System Repos]
    end
    
    subgraph "C++ Backend"
        CppBackend[C++ CUDA Accelerator<br/>gRPC Server]
    end
    
    Browser --> AuxHTTP
    Browser --> WebRTC
    Browser --> ConnectRPC
    Browser --> Vanguard
    API --> ConnectRPC
    API --> Vanguard
    
    ConnectRPC --> ProcessImageUC
    ConnectRPC --> ConfigUC
    ConnectRPC --> FileUC
    WebRTC --> ProcessImageUC
    Vanguard --> ProcessImageUC
    
    ProcessImageUC --> ImageProcessor
    CapabilitiesUC --> ImageProcessor
    ConfigUC --> FeatureFlag
    FileUC --> Image
    FileUC --> Video
    
    ImageProcessor --> GRPCProcessor
    GRPCProcessor --> CppBackend
    
    ConfigUC --> FeatureFlags
    FileUC --> FileSystem
```

### Directory Structure

```
webserver/
├── cmd/server/          # Main entry point (main.go)
├── pkg/
│   ├── application/     # Use cases (business logic)
│   │   ├── process_image_use_case.go
│   │   ├── processor_capabilities_use_case.go
│   │   ├── evaluate_feature_flag_use_case.go
│   │   ├── get_system_info_use_case.go
│   │   ├── list_available_images_use_case.go
│   │   ├── upload_image_use_case.go
│   │   ├── list_videos_use_case.go
│   │   ├── upload_video_use_case.go
│   │   ├── stream_video_use_case.go
│   │   ├── video_playback_use_case.go
│   │   └── list_inputs_use_case.go
│   ├── domain/          # Domain models and interfaces
│   │   ├── image.go
│   │   ├── processor.go
│   │   ├── video.go
│   │   ├── video_player.go
│   │   ├── feature_flag.go
│   │   ├── system_info.go
│   │   ├── device_status.go
│   │   ├── processing_options.go
│   │   └── interfaces/
│   ├── infrastructure/  # External integrations
│   │   ├── processor/   # C++/CUDA integration
│   │   │   ├── grpc_processor.go
│   │   │   ├── grpc_client.go
│   │   │   └── grpc_repository.go
│   │   ├── featureflags/# Goff integration (YAML-based)
│   │   ├── filesystem/  # File repositories
│   │   ├── video/       # Video repositories
│   │   ├── webrtc/      # WebRTC peer management
│   │   ├── mqtt/        # MQTT device monitoring
│   │   ├── http/        # HTTP utilities
│   │   ├── image/       # Image codec
│   │   ├── logger/      # Structured logging
│   │   ├── config/      # Config repository
│   │   ├── version/     # Version info
│   │   └── build/       # Build info
│   ├── interfaces/      # HTTP/Connect-RPC handlers
│   │   ├── connectrpc/  # Connect-RPC handlers
│   │   │   ├── handler.go
│   │   │   ├── config_handler.go
│   │   │   ├── file_handler.go
│   │   │   ├── webrtc_handler.go
│   │   │   ├── remote_management_handler.go
│   │   │   └── vanguard.go
│   │   ├── http/        # HTTP handlers
│   │   │   ├── health_handler.go
│   │   │   ├── logs_proxy.go
│   │   │   └── trace_proxy.go
│   │   └── adapters/   # Protocol adapters
│   ├── config/          # Configuration management
│   ├── container/       # Dependency injection
│   ├── app/             # Application setup
│   └── telemetry/       # OpenTelemetry integration
```

Frontend source: `../front-end/` (Vite in development, embedded static assets in production).

## Key Components

### Interfaces Layer

**Connect-RPC Handlers** (`pkg/interfaces/connectrpc/`):
- `handler.go`: Main image processor handler implementing Connect-RPC service interface
- `config_handler.go`: Configuration and system info handler
- `file_handler.go`: File upload and listing handler
- `webrtc_handler.go`: WebRTC signaling handler
- `remote_management_handler.go`: Remote device management (Jetson Nano)
- `vanguard.go`: REST API transcoder using Vanguard for google.api.http annotations

**HTTP Handlers** (`pkg/interfaces/http/`):
- `health_handler.go`: Health check endpoints
- `logs_proxy.go`: Loki logs proxy
- `trace_proxy.go`: Jaeger trace proxy

Static `/data/` assets are served by the Go server in both development and production.

### Application Layer

**Use Cases** (`pkg/application/`):
- `ProcessImageUseCase`: Orchestrates image processing business logic
- `ProcessorCapabilitiesUseCase`: Queries available filters and accelerators
- `EvaluateFeatureFlagUseCase`: Evaluates feature flags from Goff YAML configuration
- `GetSystemInfoUseCase`: Retrieves system information and build details
- `ListAvailableImagesUseCase`: Lists available static images
- `UploadImageUseCase`: Handles image uploads
- `ListVideosUseCase` / `UploadVideoUseCase`: Video management
- `StreamVideoUseCase`: Streams video frames via WebRTC for real-time processing
- `VideoPlaybackUseCase`: Manages video playback sessions
- `ListInputsUseCase`: Lists available input sources

All use cases follow the same pattern: they receive domain models, orchestrate business logic, and return domain models or errors.

### Domain Layer

**Domain Models** (`pkg/domain/`):
- `Image`: Core image domain model with data, dimensions, format
- `Processor`: ImageProcessor interface defining processing contract
- `Video`: Video domain model and repository interface
- `VideoPlayer`: Video player interface for playback management
- `FeatureFlag`: Feature flag domain model
- `SystemInfo`: System information and build details
- `DeviceStatus`: Device monitoring status
- `ProcessingOptions`: Image/video processing configuration options

**Repository Interfaces** (`pkg/domain/interfaces/`):
- Define contracts for data access without implementation details
- Enable dependency inversion (domain depends on abstractions, not implementations)

### Infrastructure Layer

**Processor Integration** (`pkg/infrastructure/processor/`):
- `GRPCProcessor`: Integrates with C++ library via gRPC
- `grpc_client.go`: gRPC client implementation for remote processing
- Implements `domain.ImageProcessor` interface

**Feature Flags** (`pkg/infrastructure/featureflags/`):
- Goff client integration for feature flag management (YAML-based)
- Repository pattern for feature flag evaluation
- Local YAML file configuration (not Flipt server)

**File System** (`pkg/infrastructure/filesystem/`):
- Static image repository implementation
- File-based storage for uploaded images

**Video** (`pkg/infrastructure/video/`):
- Video repository implementation
- FFmpeg integration for video processing
- Preview generation for video files

### Dependency Injection

**Container** (`pkg/container/`):
- Centralized dependency injection
- Creates and wires all components
- Manages lifecycle of use cases, repositories, and connectors

**App** (`pkg/app/`):
- Application setup and HTTP server configuration
- Registers all handlers and middleware
- Configures routing (Connect-RPC, REST via Vanguard, WebRTC signaling, static files)

## Features

- **CUDA Acceleration**: GPU-powered image processing via gRPC remote service
- **Connect-RPC**: Type-safe RPC with HTTP/JSON and gRPC support
- **Vanguard**: RESTful API transcoding using google.api.http annotations
- **Protocol Buffers**: Multiple proto services (config_service, file_service, image_processor_service, webrtc_signal)
- **Hot Reload**: Frontend development with Vite, Go hot reload for templates
- **Clean Architecture**: Domain → Application → Infrastructure → Interfaces layers
- **WebRTC**: Real-time video/image streaming with WebRTC signaling
- **OpenTelemetry**: Distributed tracing integration

### Initialization Flow

```mermaid
sequenceDiagram
    participant Main as main.go
    participant Container as Container
    participant Config as Config Manager
    participant GRPCClient as GRPCClient
    participant GRPCServer as gRPC Server
    participant App as App Setup
    
    Main->>Container: New(ctx, configFile)
    Container->>Config: New(configFile)
    Container->>Config: Load configuration
    
    Container->>GRPCClient: NewGRPCClient(ctx, config)
    GRPCClient->>GRPCServer: Dial gRPC server
    GRPCServer-->>GRPCClient: Connected
    
    Container->>Container: Create Use Cases
    Container->>Container: Create Repositories
    Container->>App: New(ctx, options...)
    App->>App: Register handlers
    App-->>Main: App ready
```

### Processing Flows

#### gRPC Processing Flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as Connect-RPC Handler
    participant UseCase as ProcessImageUseCase
    participant Processor as ImageProcessor
    participant GRPCProc as GRPCProcessor
    participant GRPCClient as GRPCClient
    participant CppBackend as C++ Backend
    
    Client->>Handler: ProcessImage(request)
    Handler->>Handler: Convert protobuf to domain
    Handler->>UseCase: Execute(ctx, image, filters, accelerator)
    
    UseCase->>Processor: ProcessImage(ctx, image, filters, accelerator)
    Processor->>GRPCProc: ProcessImage(ctx, image, filters, accelerator)
    
    GRPCProc->>GRPCProc: Convert domain to protobuf
    GRPCProc->>GRPCClient: ProcessImage(ctx, request)
    GRPCClient->>CppBackend: gRPC call ProcessImage()
    CppBackend-->>GRPCClient: ProcessImageResponse
    GRPCClient-->>GRPCProc: ProcessImageResponse
    GRPCProc->>GRPCProc: Convert protobuf to domain.Image
    GRPCProc-->>Processor: domain.Image
    Processor-->>UseCase: domain.Image
    UseCase-->>Handler: domain.Image
    Handler->>Handler: Convert domain to protobuf
    Handler-->>Client: ProcessImageResponse
```

### Endpoint Sequence Diagrams

#### ListFilters

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as ImageProcessorHandler
    participant CapabilitiesUC as ProcessorCapabilitiesUseCase
    participant Processor as ImageProcessor
    participant GRPCServer as gRPC Server
    
    Client->>Handler: ListFilters(request)
    Handler->>CapabilitiesUC: GetCapabilities(ctx)
    CapabilitiesUC->>Processor: GetCapabilities()
    Processor->>GRPCServer: gRPC ListFilters()
    GRPCServer-->>Processor: ListFiltersResponse
    Processor-->>CapabilitiesUC: Filter definitions
    CapabilitiesUC-->>Handler: Filter definitions
    Handler->>Handler: Build ListFiltersResponse
    Handler-->>Client: ListFiltersResponse
```

#### GetStreamConfig

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as ConfigHandler
    participant FFUseCase as EvaluateFeatureFlagUseCase
    participant Config as Config Manager
    
    Client->>Handler: GetStreamConfig(request)
    Handler->>Config: Read signaling endpoint
    Config-->>Handler: WebRTC endpoint
    
    Handler->>FFUseCase: EvaluateVariant("frontend_log_level")
    FFUseCase-->>Handler: logLevel
    
    Handler->>FFUseCase: EvaluateBoolean("frontend_console_logging")
    FFUseCase-->>Handler: consoleLogging
    
    Handler->>Handler: Build StreamEndpoints
    Handler-->>Client: GetStreamConfigResponse
```

#### EvaluateFeatureFlag

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as ConfigHandler
    participant EvalUC as EvaluateFeatureFlagUseCase
    participant GoffRepo as GoffRepository
    participant YAML as YAML File
    
    Client->>Handler: EvaluateFeatureFlag(request)
    Handler->>EvalUC: EvaluateString/EvaluateBoolean(ctx, flagKey)
    EvalUC->>GoffRepo: GetFlag(flagKey)
    GoffRepo->>YAML: Read configuration
    YAML-->>GoffRepo: Flag data
    GoffRepo-->>EvalUC: Flag value
    EvalUC-->>Handler: Flag evaluation result
    Handler-->>Client: EvaluateFeatureFlagResponse
```

#### GetSystemInfo

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as ConfigHandler
    participant SystemInfoUC as GetSystemInfoUseCase
    participant CapabilitiesUC as ProcessorCapabilitiesUseCase
    participant GRPCServer as gRPC Server
    participant BuildRepo as BuildInfoRepository
    
    Client->>Handler: GetSystemInfo(request)
    Handler->>SystemInfoUC: Execute(ctx)
    SystemInfoUC->>BuildRepo: GetBuildInfo()
    BuildRepo-->>SystemInfoUC: BuildInfo
    
    SystemInfoUC->>CapabilitiesUC: GetCapabilities(ctx)
    CapabilitiesUC->>GRPCServer: gRPC GetVersionInfo()
    GRPCServer-->>CapabilitiesUC: VersionInfoResponse
    CapabilitiesUC-->>SystemInfoUC: Version
    
    SystemInfoUC-->>Handler: SystemInfo
    Handler-->>Client: GetSystemInfoResponse
```

#### ListAvailableImages

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as FileHandler
    participant ListUC as ListAvailableImagesUseCase
    participant ImageRepo as ImageRepository
    participant FileSystem as File System
    
    Client->>Handler: ListAvailableImages(request)
    Handler->>ListUC: Execute(ctx)
    ListUC->>ImageRepo: ListImages()
    ImageRepo->>FileSystem: Read directory
    FileSystem-->>ImageRepo: File list
    ImageRepo->>ImageRepo: Build image metadata
    ImageRepo-->>ListUC: []Image
    ListUC-->>Handler: []Image
    Handler->>Handler: Convert to protobuf
    Handler-->>Client: ListAvailableImagesResponse
```

#### UploadImage

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as FileHandler
    participant UploadUC as UploadImageUseCase
    participant ImageRepo as ImageRepository
    participant FileSystem as File System
    
    Client->>Handler: UploadImage(request)
    Handler->>Handler: Validate request
    Handler->>UploadUC: Execute(ctx, filename, fileData)
    UploadUC->>UploadUC: Validate file size/format
    UploadUC->>ImageRepo: SaveImage(filename, data)
    ImageRepo->>FileSystem: Write file
    FileSystem-->>ImageRepo: Success
    ImageRepo->>ImageRepo: Generate image metadata
    ImageRepo-->>UploadUC: Image
    UploadUC-->>Handler: Image
    Handler->>Handler: Convert to protobuf
    Handler-->>Client: UploadImageResponse
```

#### ListVideos

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as FileHandler
    participant ListUC as ListVideosUseCase
    participant VideoRepo as VideoRepository
    participant FileSystem as File System
    
    Client->>Handler: ListAvailableVideos(request)
    Handler->>ListUC: Execute(ctx)
    ListUC->>VideoRepo: ListVideos()
    VideoRepo->>FileSystem: Read directory
    FileSystem-->>VideoRepo: File list
    VideoRepo->>VideoRepo: Build video metadata
    VideoRepo-->>ListUC: []Video
    ListUC-->>Handler: []Video
    Handler->>Handler: Convert to protobuf
    Handler-->>Client: ListAvailableVideosResponse
```

#### UploadVideo

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as FileHandler
    participant UploadUC as UploadVideoUseCase
    participant VideoRepo as VideoRepository
    participant FileSystem as File System
    participant PreviewGen as Preview Generator
    
    Client->>Handler: UploadVideo(request)
    Handler->>Handler: Validate request
    Handler->>UploadUC: Execute(ctx, fileData, filename)
    UploadUC->>UploadUC: Validate file size/format
    UploadUC->>VideoRepo: SaveVideo(filename, data)
    VideoRepo->>FileSystem: Write file
    FileSystem-->>VideoRepo: Success
    VideoRepo->>PreviewGen: GeneratePreview(videoPath)
    PreviewGen->>PreviewGen: Extract frame with FFmpeg
    PreviewGen-->>VideoRepo: Preview image path
    VideoRepo->>VideoRepo: Generate video metadata
    VideoRepo-->>UploadUC: Video
    UploadUC-->>Handler: Video
    Handler->>Handler: Convert to protobuf
    Handler-->>Client: UploadVideoResponse
```

## Protocol Buffers

The project uses multiple proto service definitions:

- `proto/config_service.proto` - Configuration and system info (with REST annotations)
- `proto/file_service.proto` - File upload and listing (with REST annotations)
- `proto/image_processor_service.proto` - Image processing operations (with REST annotations)
- `proto/webrtc_signal.proto` - WebRTC signaling service
- `proto/remote_management_service.proto` - Remote device management
- `proto/common.proto` - Shared message types

All services include `google.api.http` annotations for RESTful routing via Vanguard transcoder.

Generate code:
```bash
./scripts/build/protos.sh
# Or manually:
docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
```

## Frontend

The frontend uses:
- **Lit Web Components** - Native web components
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server
- **Vitest** - Unit testing
- **Playwright** - E2E testing

**Development:**
```bash
cd ../front-end
npm install
npm run dev  # Vite dev server (full stack: ../scripts/dev/start.sh)
```

**Production:**
In production, the frontend is built with Vite and embedded as static assets in the Go binary. No separate Nginx server is needed.

```bash
cd ../front-end && npm run build
```

## See Also

- [Main README](../README.md) - Project overview and setup
- [Testing Documentation](../docs/testing-and-coverage.md) - Test execution guide
