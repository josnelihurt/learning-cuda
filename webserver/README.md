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
2. Start services (Flipt, Jaeger, etc.)
3. Enable hot reload for frontend (Vite dev server)
4. Start the Go server with hot reload support

**Access:** https://localhost:8443

### Manual Build

Build the Go server:
```bash
cd webserver
make build
```

Or from project root:
```bash
cd webserver && make build
```

Run the server:
```bash
cd webserver
make run
```

Or with config file:
```bash
./bin/server -config=config/config.yaml
```

### Development Mode

Run server with hot reload:
```bash
cd webserver
make dev
```

## Production Mode

For production (embedded files, single binary):

```bash
cd webserver
make build
./bin/server -config=config/config.production.yaml
```

The frontend is built with Vite and embedded as static assets. Templates and static files are served from the binary.

## Architecture

The web server implements Clean Architecture with clear separation of concerns across four main layers: Interfaces, Application, Domain, and Infrastructure.

### Component Overview

The web server follows Clean Architecture principles with clear separation between interfaces, application logic, domain models, and infrastructure. It integrates with the C++ CUDA accelerator library via CGO (shared library) or gRPC.

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
        API[API Clients]
    end
    
    subgraph "Go Interfaces Layer"
        ConnectRPC[Connect-RPC Handlers]
        WebSocket[WebSocket Handlers]
        StaticHTTP[Static HTTP Handler]
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
        CppConnector[CppConnector]
        GRPCProcessor[GRPCProcessor]
        Loader[Loader - dlopen]
        FeatureFlags[Flipt Integration]
        FileSystem[File System Repos]
    end
    
    subgraph "C++ Backend"
        CppBackend[C++ CUDA Accelerator<br/>gRPC / CGO]
    end
    
    Browser --> StaticHTTP
    Browser --> WebSocket
    API --> ConnectRPC
    API --> Vanguard
    
    ConnectRPC --> ProcessImageUC
    ConnectRPC --> ConfigUC
    ConnectRPC --> FileUC
    WebSocket --> ProcessImageUC
    Vanguard --> ProcessImageUC
    
    ProcessImageUC --> ImageProcessor
    CapabilitiesUC --> ImageProcessor
    ConfigUC --> FeatureFlag
    FileUC --> Image
    FileUC --> Video
    
    ImageProcessor --> CppConnector
    ImageProcessor --> GRPCProcessor
    CppConnector --> Loader
    Loader --> CppBackend
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
│   │   ├── get_stream_config_use_case.go
│   │   ├── sync_feature_flags_use_case.go
│   │   ├── list_available_images_use_case.go
│   │   └── upload_image_use_case.go
│   ├── domain/          # Domain models and interfaces
│   │   ├── image.go
│   │   ├── processor.go
│   │   ├── video.go
│   │   ├── feature_flag.go
│   │   └── interfaces/
│   ├── infrastructure/  # External integrations
│   │   ├── processor/   # C++/CUDA integration
│   │   │   ├── cpp_connector.go
│   │   │   ├── grpc_processor.go
│   │   │   ├── grpc_client.go
│   │   │   └── loader/  # dlopen wrapper
│   │   ├── featureflags/# Flipt integration
│   │   ├── filesystem/  # File repositories
│   │   ├── video/       # Video repositories
│   │   └── build/       # Build info repository
│   ├── interfaces/      # HTTP/WebSocket/Connect-RPC handlers
│   │   ├── connectrpc/  # Connect-RPC handlers
│   │   ├── websocket/   # WebSocket handlers
│   │   ├── statichttp/  # Static file serving
│   │   └── adapters/   # Protocol adapters
│   ├── config/          # Configuration management
│   ├── container/       # Dependency injection
│   ├── app/             # Application setup
│   └── telemetry/       # OpenTelemetry integration
└── web/
    ├── src/             # TypeScript source (Lit Web Components)
    ├── templates/       # HTML templates
    └── static/          # Static assets (CSS, images)
```

## Key Components

### Interfaces Layer

**Connect-RPC Handlers** (`pkg/interfaces/connectrpc/`):
- `handler.go`: Main image processor handler implementing Connect-RPC service interface
- `config_handler.go`: Configuration and system info handler
- `file_handler.go`: File upload and listing handler
- `vanguard.go`: REST API transcoder using Vanguard for google.api.http annotations

**WebSocket Handler** (`pkg/interfaces/websocket/`):
- Real-time bidirectional communication for video/image processing
- Session management for multiple concurrent connections
- Streams processing results back to clients

**Static HTTP Handler** (`pkg/interfaces/statichttp/`):
- Serves frontend SPA (Single Page Application)
- Handles development mode (Vite proxy) and production mode (embedded assets)
- Template rendering and static asset serving

### Application Layer

**Use Cases** (`pkg/application/`):
- `ProcessImageUseCase`: Orchestrates image processing business logic
- `ProcessorCapabilitiesUseCase`: Queries available filters and accelerators
- `GetStreamConfigUseCase`: Retrieves streaming configuration
- `SyncFeatureFlagsUseCase`: Synchronizes feature flags from Flipt
- `ListAvailableImagesUseCase`: Lists available static images
- `UploadImageUseCase`: Handles image uploads
- `ListVideosUseCase` / `UploadVideoUseCase`: Video management

All use cases follow the same pattern: they receive domain models, orchestrate business logic, and return domain models or errors.

### Domain Layer

**Domain Models** (`pkg/domain/`):
- `Image`: Core image domain model with data, dimensions, format
- `Processor`: ImageProcessor interface defining processing contract
- `Video`: Video domain model and repository interface
- `FeatureFlag`: Feature flag domain model

**Repository Interfaces** (`pkg/domain/interfaces/`):
- Define contracts for data access without implementation details
- Enable dependency inversion (domain depends on abstractions, not implementations)

### Infrastructure Layer

**Processor Integration** (`pkg/infrastructure/processor/`):
- `CppConnector`: Integrates with C++ library via CGO (dlopen)
- `GRPCProcessor`: Integrates with C++ library via gRPC
- `Loader`: Wraps dlopen functionality for dynamic library loading
- Both connectors implement `domain.ImageProcessor` interface

**Feature Flags** (`pkg/infrastructure/featureflags/`):
- Flipt client integration for feature flag management
- Repository pattern for feature flag persistence
- HTTP API client for Flipt server communication

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
- Configures routing (Connect-RPC, REST via Vanguard, WebSocket, static files)

## Features

- **CUDA Acceleration**: GPU-powered image processing via dynamic plugin system (dlopen/CGO) or gRPC remote service
- **Connect-RPC**: Type-safe RPC with HTTP/JSON and gRPC support
- **Vanguard**: RESTful API transcoding using google.api.http annotations
- **Protocol Buffers**: Multiple proto services (config_service, file_service, image_processor_service)
- **Hot Reload**: Frontend development with Vite, Go hot reload for templates
- **Clean Architecture**: Domain → Application → Infrastructure → Interfaces layers
- **WebSocket**: Real-time video/image processing
- **OpenTelemetry**: Distributed tracing integration

### Initialization Flow

```mermaid
sequenceDiagram
    participant Main as main.go
    participant Container as Container
    participant Config as Config Manager
    participant CppConn as CppConnector
    participant Loader as Loader
    participant CppBackend as C++ Backend
    participant GRPCClient as GRPCClient
    participant App as App Setup
    
    Main->>Container: New(ctx, configFile)
    Container->>Config: New(configFile)
    Container->>Config: Load configuration
    
    alt CGO Mode
        Container->>CppConn: New(libraryPath)
        CppConn->>Loader: NewLoader(path)
        Loader->>Loader: dlopen(libcuda_processor.so)
        Loader->>CppBackend: processor_init()
        CppBackend-->>Loader: InitResponse
        Loader-->>CppConn: Ready
    else gRPC Mode
        Container->>GRPCClient: NewGRPCClient(ctx, config)
        GRPCClient->>GRPCClient: Dial gRPC server
        GRPCClient-->>Container: Connected
    end
    
    Container->>Container: Create Use Cases
    Container->>Container: Create Repositories
    Container->>App: New(ctx, options...)
    App->>App: Register handlers
    App-->>Main: App ready
```

### Processing Flows

#### CGO (Shared Library) Processing Flow

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as Connect-RPC Handler
    participant UseCase as ProcessImageUseCase
    participant Processor as ImageProcessor
    participant CppConn as CppConnector
    participant Loader as Loader
    participant CppBackend as C++ Backend
    
    Client->>Handler: ProcessImage(request)
    Handler->>Handler: Convert protobuf to domain
    Handler->>UseCase: Execute(ctx, image, filters, accelerator)
    
    UseCase->>Processor: ProcessImage(ctx, image, filters, accelerator)
    Processor->>CppConn: ProcessImage(ctx, image, filters, accelerator)
    
    CppConn->>CppConn: Convert domain to protobuf
    CppConn->>Loader: ProcessImage(ProcessImageRequest)
    Loader->>Loader: Marshal protobuf
    Loader->>CppBackend: processor_process_image()
    CppBackend-->>Loader: ProcessImageResponse
    Loader->>Loader: Unmarshal protobuf
    Loader-->>CppConn: ProcessImageResponse
    CppConn->>CppConn: Convert protobuf to domain.Image
    CppConn-->>Processor: domain.Image
    Processor-->>UseCase: domain.Image
    UseCase-->>Handler: domain.Image
    Handler->>Handler: Convert domain to protobuf
    Handler-->>Client: ProcessImageResponse
```

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
    participant FFUseCase as EvaluateFeatureFlagUseCase
    participant CapabilitiesUC as ProcessorCapabilitiesUseCase
    participant Processor as ImageProcessor
    participant CppBackend as C++ Backend
    
    Client->>Handler: ListFilters(request)
    Handler->>FFUseCase: EvaluateBoolean("processor_use_grpc_backend")
    FFUseCase-->>Handler: useGRPC flag
    
    alt gRPC Mode
        Handler->>CapabilitiesUC: GetCapabilities(ctx)
        CapabilitiesUC->>Processor: GetCapabilities()
        Processor->>CppBackend: gRPC GetCapabilities()
        CppBackend-->>Processor: CapabilitiesResponse
        Processor-->>CapabilitiesUC: Capabilities
    else CGO Mode
        Handler->>CapabilitiesUC: GetCapabilities(ctx)
        CapabilitiesUC->>Processor: GetCapabilities()
        Processor->>CppBackend: processor_get_capabilities()
        CppBackend-->>Processor: CapabilitiesResponse
        Processor-->>CapabilitiesUC: Capabilities
    end
    
    CapabilitiesUC-->>Handler: Filter definitions
    Handler->>Handler: Build ListFiltersResponse
    Handler-->>Client: ListFiltersResponse
```

#### GetStreamConfig

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as ConfigHandler
    participant StreamConfigUC as GetStreamConfigUseCase
    participant FFUseCase as EvaluateFeatureFlagUseCase
    participant Config as Config Manager
    
    Client->>Handler: GetStreamConfig(request)
    Handler->>StreamConfigUC: Execute(ctx)
    StreamConfigUC->>Config: GetStreamConfig()
    Config-->>StreamConfigUC: StreamConfig
    
    Handler->>FFUseCase: EvaluateVariant("frontend_log_level")
    FFUseCase-->>Handler: logLevel
    
    Handler->>FFUseCase: EvaluateBoolean("frontend_console_logging")
    FFUseCase-->>Handler: consoleLogging
    
    Handler->>Handler: Build StreamEndpoints
    Handler-->>Client: GetStreamConfigResponse
```

#### SyncFeatureFlags

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as ConfigHandler
    participant SyncUC as SyncFeatureFlagsUseCase
    participant FFRepo as FeatureFlagRepository
    participant Flipt as Flipt Server
    
    Client->>Handler: SyncFeatureFlags(request)
    Handler->>Handler: Build feature flag definitions
    Handler->>SyncUC: Execute(ctx, flags)
    SyncUC->>FFRepo: SyncFlags(flags)
    FFRepo->>Flipt: Create/Update flags
    Flipt-->>FFRepo: Success
    FFRepo-->>SyncUC: Success
    SyncUC-->>Handler: Success
    Handler-->>Client: SyncFeatureFlagsResponse
```

#### GetSystemInfo

```mermaid
sequenceDiagram
    participant Client as Client
    participant Handler as ConfigHandler
    participant SystemInfoUC as GetSystemInfoUseCase
    participant CppConn as CppConnector
    participant CppBackend as C++ Backend
    participant BuildRepo as BuildInfoRepository
    
    Client->>Handler: GetSystemInfo(request)
    Handler->>SystemInfoUC: Execute(ctx)
    SystemInfoUC->>BuildRepo: GetBuildInfo()
    BuildRepo-->>SystemInfoUC: BuildInfo
    
    SystemInfoUC->>CppConn: GetVersionInfo()
    CppConn->>CppBackend: processor_get_library_version()
    CppBackend-->>CppConn: Version string
    CppConn-->>SystemInfoUC: Version
    
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
cd webserver/web
npm install
npm run dev  # Vite dev server
```

**Build:**
```bash
npm run build
```

## See Also

- [Main README](../README.md) - Project overview and setup
- [Testing Documentation](../docs/testing-and-coverage.md) - Test execution guide
