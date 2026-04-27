# CUDA Accelerator Library

High-performance image processing library implementing Clean Architecture principles with CUDA GPU acceleration and CPU fallback support.

## Library Description

The CUDA Accelerator Library provides a production-grade image processing framework with GPU-accelerated filters using CUDA kernels. The architecture follows Clean Architecture patterns with clear separation between domain logic, application use cases, infrastructure implementations, and external adapters.

**Version**: See `VERSION` file (currently 4.0.2)

**Note**: The library version (4.0.2) is separate from the C API version (2.1.0 defined in `processor_api.h`). The API version indicates the C interface contract, while the library version tracks overall library releases.

**Features**:
- GPU acceleration via CUDA kernels with CPU fallback
- **Accelerator Control Client** with mTLS outbound connections to Go cloud server
- **Multiplexed bidirectional gRPC stream** for all commands (image processing, filters, version, signaling)
- WebRTC signaling support for real-time video streaming
- **YOLO object detection** via TensorRT with GPU-accelerated inference
- **Data channel framing** for structured detection result transport over WebRTC
- Extensible filter pipeline architecture
- Thread-safe concurrent processing
- Buffer pooling and CUDA memory pooling for memory efficiency
- Configuration management system

## Architecture

### Component Overview

The library uses the **Accelerator Control Client** as the primary integration path. The client dials outbound to a Go cloud server via mTLS and establishes a multiplexed bidirectional stream used exclusively for registration, WebRTC signaling, and keepalives. All image processing occurs over **WebRTC peer connections** established through that signaling. The Go server negotiates WebRTC sessions; once connected, video frames flow over RTP media tracks while still-image processing, detection results, and control requests flow over dedicated data channels. All processing converges at the `ProcessorEngine` which orchestrates image processing through the filter pipeline.

```mermaid
graph TB
    subgraph "Go Cloud Server"
        GoServer[Go Server<br/>AcceleratorControlService<br/>mTLS]
        Registry[Registry<br/>Session Manager]
    end

    subgraph "Accelerator Control Client (C++)"
        AccelClient[AcceleratorControlClient<br/>Outbound mTLS]
        BidiStream[Bidi Stream<br/>Signaling + Keepalive]
    end

    subgraph "WebRTC Sessions"
        WebRTCMgr[WebRTCManager]
        LiveVP[LiveVideoProcessor<br/>H264 decode/encode]
        DataChannels[Data Channels<br/>control / detections / stats]
        MediaTracks[Media Tracks<br/>inbound/outbound H264 RTP]
    end

    subgraph "C++ Core Processing"
        ProcessorEngine[ProcessorEngine]
        FilterPipeline[FilterPipeline]
        BufferPool[BufferPool]
    end

    subgraph "Domain Interfaces"
        IFilter[IFilter]
        ImageBuffer[ImageBuffer]
    end

    subgraph "Infrastructure"
        CudaFilters[CUDA Filters]
        CpuFilters[CPU Filters]
        YOLODetector[YOLO Detector<br/>TensorRT]
        ModelManager[Model Manager]
    end

    GoServer --> Registry

    AccelClient -->|mTLS signaling| GoServer
    AccelClient --> BidiStream
    AccelClient --> WebRTCMgr

    WebRTCMgr --> LiveVP
    WebRTCMgr --> DataChannels
    WebRTCMgr --> MediaTracks
    LiveVP --> ProcessorEngine

    ProcessorEngine --> FilterPipeline
    FilterPipeline --> BufferPool
    FilterPipeline --> IFilter
    IFilter --> CudaFilters
    IFilter --> CpuFilters
    CudaFilters --> YOLODetector
    YOLODetector --> ModelManager
    FilterPipeline --> ImageBuffer
```

### Layer Structure

```mermaid
graph TB
    subgraph "Go Cloud Server"
        GoServer[Go Server<br/>AcceleratorControlService]
    end

    subgraph "C++ Ports Layer"
        AccelClient[AcceleratorControlClient]
        Provider[ProcessorEngineAdapter]
        WebRTCPort[WebRTCManager]
        DataChannelFraming[DataChannelFraming]
        LiveVideoProcessor[LiveVideoProcessor]
    end

    subgraph "C++ Application Layer"
        ProcessorEngine[ProcessorEngine]
        FilterPipeline[FilterPipeline]
        BufferPool[BufferPool]
        CommandFactory[CommandFactory]
    end

    subgraph "C++ Domain Layer"
        IFilter[IFilter Interface]
        ImageBuffer[ImageBuffer]
        IImageProcessor[IImageProcessor]
        GrayscaleAlgorithm[GrayscaleAlgorithm]
    end

    subgraph "C++ Infrastructure Layer"
        CudaFilters[CUDA Filters]
        CpuFilters[CPU Filters]
        ImageIO[Image I/O]
        ConfigManager[ConfigManager]
    end

    subgraph "C++ Core Layer"
        Logger[Logger]
        Telemetry[Telemetry]
        Result[Result Type]
    end

    GoServer -->|mTLS bidi| AccelClient
    AccelClient --> Provider
    Provider --> ProcessorEngine
    AccelClient --> WebRTCPort

    ProcessorEngine --> FilterPipeline
    FilterPipeline --> BufferPool
    FilterPipeline --> IFilter
    IFilter --> CudaFilters
    IFilter --> CpuFilters
    CommandFactory --> ConfigManager
    ProcessorEngine --> Logger
    ProcessorEngine --> Telemetry
```

### Initialization Sequence

```mermaid
sequenceDiagram
    participant Main as main_client.cpp
    participant Flags as absl Flags
    participant Engine as ProcessorEngine
    participant TM as TelemetryManager
    participant Provider as ProcessorEngineAdapter
    participant WebRTC as WebRTCManager
    participant Client as AcceleratorControlClient

    Main->>Flags: Parse flags (control_addr, device_id, mTLS paths)
    Flags-->>Main: Configuration

    Main->>Engine: Create ProcessorEngine("accelerator-client")
    Main->>Engine: Initialize(InitRequest)
    Engine->>TM: Initialize("accelerator-client", endpoint)
    TM->>TM: Create OpenTelemetry exporter
    Engine-->>Main: InitResponse{code: 0}

    Main->>Provider: Create ProcessorEngineAdapter(engine)
    Main->>WebRTC: Create WebRTCManager
    Main->>Client: Create AcceleratorControlClient(config, provider, webrtc)
    Main->>Client: Run()

    Client->>Client: Load mTLS credentials
    Client->>Client: Create mTLS gRPC channel
    Client->>Client: Dial Go control server
    Client->>Client: Connect() - open bidi stream
    Client->>Client: Send Register(device_id, caps, version)
    Client-->>Client: Receive RegisterAck(session_id)

    Note over Client: Connected and registered<br/>Ready to receive commands

    loop Reconnect with backoff
        Client->>Client: Read/Dispatch loop
        Note over Client: SignalingMessage, Keepalive
    end
```

### Processing Flows

The gRPC bidi stream carries only **signaling** and **keepalive** messages. All image processing flows through WebRTC peer connections after session establishment.

#### Live Video Processing (WebRTC Media Track)

The primary path for real-time video. Go sends H.264 video via RTP; C++ decodes, processes, re-encodes, and sends processed video back on an outbound track.

```mermaid
sequenceDiagram
    participant Browser as Browser (via Go)
    participant Go as Go Server
    participant Client as AcceleratorControlClient
    participant WRTC as WebRTCManager
    participant LVP as LiveVideoProcessor
    participant Engine as ProcessorEngine
    participant Filter as Filter (CUDA/CPU)

    Note over Go,Client: 1. Signaling via gRPC bidi stream
    Go->>Client: SignalingMessage(StartSession, SDP offer)
    Client->>WRTC: CreateSession(session_id, sdp_offer)
    WRTC-->>Client: SDP answer + ICE candidates
    Client->>Go: SignalingMessage(SDP answer + local candidates)
    Go->>Client: SignalingMessage(remote ICE candidates)
    Client->>WRTC: HandleRemoteCandidate(...)
    Note over Browser,WRTC: WebRTC peer connection established

    Note over Browser,WRTC: 2. Video frames via RTP (not gRPC)
    Browser->>WRTC: H.264 RTP video frames (inbound track)
    WRTC->>LVP: ProcessAccessUnit(H264 AU)
    LVP->>LVP: FFmpeg decode → RGB
    LVP->>Engine: ProcessImage(RGB request)
    Engine->>Filter: Apply filters (CUDA/CPU)
    Filter-->>Engine: Processed image
    Engine-->>LVP: ProcessImageResponse
    LVP->>LVP: RGB → FFmpeg encode → H.264 NALUs
    WRTC->>WRTC: Send framed detections on "detections" data channel
    WRTC->>WRTC: Send ProcessingStatsFrame on "cpp-processor-stats" channel
    WRTC->>Browser: H.264 RTP processed video (outbound track)
```

#### Still Image Processing (WebRTC Data Channel)

For individual frame processing requests sent as protobuf over the default data channel.

```mermaid
sequenceDiagram
    participant Peer as Remote Peer (via Go)
    participant WRTC as WebRTCManager
    participant Engine as ProcessorEngine
    participant Filter as Filter (CUDA/CPU)

    Peer->>WRTC: ProcessImageRequest (framed binary on data channel)
    WRTC->>WRTC: Reassemble chunks via ChunkReassembler
    WRTC->>WRTC: Parse ProcessImageRequest protobuf

    alt Control update (no image data)
        WRTC->>WRTC: UpdateFilterState on LiveVideoProcessor
    else Full image
        WRTC->>Engine: ProcessImage(request, response)
        Engine->>Filter: Apply filters (CUDA/CPU)
        Filter-->>Engine: Processed image
        Engine-->>WRTC: ProcessImageResponse
        WRTC->>Peer: ProcessImageResponse (framed binary on data channel)
        WRTC->>Peer: ProcessingStatsFrame (on "cpp-processor-stats" channel)
        opt Detections found
            WRTC->>Peer: DetectionFrame (on "detections" channel)
        end
    end
```

#### Control Requests (WebRTC Data Channel)

Filter discovery and version queries are handled on a dedicated "control" data channel.

```mermaid
sequenceDiagram
    participant Peer as Remote Peer (via Go)
    participant WRTC as WebRTCManager
    participant Engine as ProcessorEngine

    Peer->>WRTC: ControlRequest(kListFilters) on "control" channel
    WRTC->>Engine: GetCapabilities()
    Engine-->>WRTC: filter definitions
    WRTC->>Peer: ControlResponse(ListFiltersResponse)

    Peer->>WRTC: ControlRequest(kGetVersion) on "control" channel
    WRTC->>WRTC: Read VERSION file + engine capabilities
    WRTC->>Peer: ControlResponse(GetVersionInfoResponse)
```

## Directory Structure

```
cpp_accelerator/
├── application/                # Application layer - use cases and orchestration
│   ├── commands/               # Command pattern implementation (placeholder)
│   │   ├── command_interface.h
│   │   ├── command_factory.h/cpp
│   │   └── command_factory_test.cpp
│   └── pipeline/               # Filter pipeline implementation
│       ├── filter_pipeline.h/cpp
│       ├── buffer_pool.h/cpp
│       └── filter_pipeline_test.cpp
├── domain/                     # Domain layer - business logic interfaces
│   └── interfaces/             # Abstraction interfaces
│       ├── filters/            # Filter interfaces
│       │   └── i_filter.h
│       ├── processors/         # Processor interfaces
│       │   └── i_image_processor.h
│       ├── image_buffer.h
│       ├── image_source.h
│       ├── image_sink.h
│       ├── i_pixel_getter.h
│       └── grayscale_algorithm.h
├── infrastructure/             # Infrastructure layer - concrete implementations
│   ├── cuda/                   # CUDA kernel implementations → [see README](infrastructure/cuda/README.md)
│   │   ├── cuda_memory_pool.h/cpp             # Thread-local GPU allocation cache
│   │   ├── grayscale_kernel.cu/h              # Grayscale CUDA kernel
│   │   ├── grayscale_filter.h/cpp             # Grayscale filter (CUDA)
│   │   ├── blur_kernel.cu/h                   # Gaussian blur CUDA kernels (4 variants)
│   │   ├── blur_processor.h/wrapper.cpp       # Blur processor host interface
│   │   ├── letterbox_kernel.cu/h              # Aspect-preserving resize + NCHW convert for TRT
│   │   ├── i_yolo_detector.h                  # YOLO detector interface (extends IFilter)
│   │   ├── yolo_detector.h/cpp                # TensorRT-based YOLO inference engine
│   │   ├── yolo_factory.h/cpp                 # Factory for creating YOLO detector instances
│   │   ├── detection.h                        # Detection result struct
│   │   ├── model_manager.h/cpp                # Model loading & management
│   │   └── model_registry.h/cpp               # Model path registry
│   ├── cpu/                    # CPU fallback implementations
│   │   ├── grayscale_filter.h/cpp
│   │   └── blur_filter.h/cpp
│   ├── image/                  # Image I/O adapters
│   │   ├── image_loader.h/cpp
│   │   └── image_writer.h/cpp
│   ├── config/                 # Configuration management
│   │   ├── config_manager.h/cpp
│   │   └── models/
│   │       └── program_config.h
│   └── filters/                # Cross-accelerator tests
│       └── blur_equivalence_test.cpp
├── ports/                      # Ports layer - external adapters
│   ├── grpc/                   # Accelerator control client (primary integration path)
│   │   ├── accelerator_control_client.h/cpp   # Outbound mTLS client to Go server
│   │   ├── processor_engine_adapter.h/cpp     # Adapter for ProcessorEngine
│   │   ├── processor_engine_provider.h        # Provider interface
│   │   ├── webrtc_manager.h/cpp               # WebRTC session management
│   │   ├── data_channel_framing.h/cpp         # Binary framing protocol for WebRTC data channels
│   │   ├── live_video_processor.h/cpp         # Real-time video frame processing
│   │   └── main_client.cpp                    # Client entry point
│   └── shared_lib/             # Shared library exports (C API)
│       ├── processor_api.h                    # C API header
│       ├── processor_engine.h/cpp             # Processor engine shared lib wrapper
│       ├── library_version.h                  # Library version constants
│       └── blur_e2e_test.cpp                  # End-to-end blur test
├── core/                       # Core utilities
│   ├── logger.h/cpp            # Logging infrastructure
│   ├── telemetry.h/cpp         # OpenTelemetry integration
│   ├── otel_log_sink.h/cpp     # OpenTelemetry log sink for spdlog
│   ├── signal_handler.h/cpp    # Process signal handling
│   └── result.h                # Error handling types
├── docker-cuda-runtime/        # CUDA runtime image for deployment
│   ├── Dockerfile
│   └── VERSION
├── yolo-model-gen/             # YOLO model generation container
│   ├── Dockerfile
│   └── VERSION
├── docker-compose.yml
├── Dockerfile.build            # Build image for gRPC server
├── Dockerfile.build.mock       # Mock build image for testing
├── VERSION                     # Library version file
└── lessons_learned.md          # Development notes and learnings
```

## Sub-folder Documentation

- **[infrastructure/cuda/README.md](infrastructure/cuda/README.md)** — Comprehensive CUDA tutorial covering kernel implementations, memory hierarchy, blur optimization variants, letterbox preprocessing, and TensorRT YOLO inference pipeline.

## Design Principles

1. **Dependency Inversion**: Domain interfaces define contracts; infrastructure implements them
2. **Single Responsibility**: Each component has one clear purpose
3. **Open/Closed**: Extend via new implementations, not modification
4. **Liskov Substitution**: All filter implementations are interchangeable
5. **Interface Segregation**: Small, focused interfaces (IFilter, ImageBuffer)
6. **Separation of Concerns**: Clear boundaries between layers
7. **Filter Pipeline**: Composable filter architecture for chaining multiple filters

## Key Components

### Accelerator Control Client

The library provides an outbound gRPC client (`AcceleratorControlClient`) that connects to a Go cloud server via mTLS. The client implements the `AcceleratorControlService` protocol buffer interface with a single multiplexed bidirectional stream.

**Multiplexed Message Types** (gRPC bidi stream):

The client sends and receives messages through the `AcceleratorMessage` envelope with `oneof payload`. The gRPC stream carries only signaling and registration; actual processing uses WebRTC.

- **Register** (C++ → Go): First message sent on connection
  - Contains `device_id`, `display_name`, `accelerator_version`, `capabilities`
  - Go server responds with `RegisterAck` (accepted/rejected, assigned `session_id`)

- **SignalingMessage** (bidirectional): WebRTC session negotiation
  - `StartSession`: Go sends SDP offer, C++ responds with SDP answer
  - `IceCandidate`: Bidirectional ICE candidate exchange
  - `CloseSession`: Session teardown

- **Keepalive** (bidirectional): Liveness check
  - No reply expected; used to detect dead connections

**WebRTC Data Channel Message Types** (after peer connection established):

- **ProcessImageRequest**: Image processing and live filter control updates (default data channel)
- **ControlRequest**: `ListFilters` and `GetVersion` queries ("control" data channel)
- **DetectionFrame**: Detection results from YOLO inference ("detections" data channel)
- **ProcessingStatsFrame**: Per-frame timing metrics ("cpp-processor-stats" data channel)

**Architecture**:

The `AcceleratorControlClient` holds a `ProcessorEngineProvider` interface for local processing and a `WebRTCManager` for WebRTC peer connections. The client:
1. Dials the Go control server with mTLS credentials
2. Opens a bidirectional stream via `Connect()`
3. Sends `Register` message with device metadata
4. Waits for `RegisterAck` confirmation
5. Enters read/dispatch loop, processing incoming commands from Go

Each message carries a `command_id` (UUID v7) for request/response correlation and an optional `trace_context` (W3C) for distributed tracing.

### WebRTC Real-time Video Processing

The library provides WebRTC-based real-time video streaming capabilities through WebRTCManager. WebRTC signaling messages are tunneled through the AcceleratorControlClient's gRPC bidi stream; after session establishment, all data flows over the peer connection.

**Components**:

- **WebRTCManager** (`ports/grpc/webrtc_manager.h/cpp`): Manages WebRTC peer connections, ICE candidate exchange, session lifecycle, and per-session CUDA memory pools
- **LiveVideoProcessor** (`ports/grpc/live_video_processor.h/cpp`): Real-time video frame processing pipeline — FFmpeg H.264 decode → RGB → ProcessorEngine → RGB → FFmpeg H.264 encode
- **DataChannelFraming** (`ports/grpc/data_channel_framing.h/cpp`): Binary chunking/reassembly protocol for large protobuf messages over SCTP data channels

**WebRTC Channels per Session**:

| Channel | Direction | Purpose |
|---|---|---|
| Inbound video track | Peer → C++ | H.264 RTP live camera frames |
| Outbound video track | C++ → Peer | H.264 RTP processed video |
| Default data channel | Bidirectional | `ProcessImageRequest`/`Response` for still images + live filter state updates |
| `control` | Bidirectional | `ListFilters`, `GetVersion` queries |
| `detections` | C++ → Peer | `DetectionFrame` with YOLO bounding boxes |
| `cpp-processor-stats` | C++ → Peer | `ProcessingStatsFrame` with per-frame timing metrics |

**Signaling Flow**:

1. Go server sends `SignalingMessage(StartSession)` with SDP offer via gRPC bidi stream
2. AcceleratorControlClient dispatches to WebRTCManager
3. WebRTCManager creates PeerConnection, sets up track/channel handlers, generates SDP answer
4. ICE candidates exchanged via `SignalingMessage(IceCandidate)` through the bidi stream
5. Direct WebRTC peer connection established — all subsequent data flows over RTP/data channels

### YOLO Object Detection

The library includes YOLO object detection via TensorRT for GPU-accelerated inference. See the [CUDA infrastructure README](infrastructure/cuda/README.md) for detailed documentation on the inference pipeline, letterbox preprocessing, and TensorRT engine lifecycle.

**Components**:

- **IYoloDetector** (`infrastructure/cuda/i_yolo_detector.h`): Detector interface extending `IFilter`
- **YOLODetector** (`infrastructure/cuda/yolo_detector.h/cpp`): TensorRT-based inference engine with NMS post-processing
- **YoloFactory** (`infrastructure/cuda/yolo_factory.h/cpp`): Factory for creating detector instances
- **ModelManager** (`infrastructure/cuda/model_manager.h/cpp`): Model loading & session management
- **ModelRegistry** (`infrastructure/cuda/model_registry.h/cpp`): Model path resolution
- **LetterboxKernel** (`infrastructure/cuda/letterbox_kernel.cu/h`): GPU-accelerated resize + pad + NCHW conversion

### Command Pattern

The command pattern infrastructure (`application/commands/`) is maintained for potential future use. All processing is currently handled directly by `FilterPipeline` which orchestrates filter chains without the command pattern abstraction layer.

### Buffer Pool

The `BufferPool` class provides efficient memory management for image processing operations by reusing allocated buffers. The buffer pool is optional — `FilterPipeline` can operate with or without it. When provided, it significantly improves performance for pipelines with multiple filters.

### Processor Engine

The `ProcessorEngine` is the core orchestration component that coordinates image processing operations. It bridges the gRPC service interface and the internal processing pipeline.

**Responsibilities**: Initialization and telemetry setup, filter orchestration via `FilterPipeline`, algorithm selection from protocol buffer enums, and response building.

**Integration Points**: Used by `ports/grpc` via `ProcessorEngineAdapter` for the gRPC service.

### Domain Interfaces

The domain layer defines core abstractions used throughout the library:

**FilterType Enum**: `GRAYSCALE`, `BLUR`

**GrayscaleAlgorithm Enum**: `BT601` (SDTV), `BT709` (HDTV), `Average`, `Lightness`, `Luminosity`

**FilterContext Structure**: Contains `ImageBuffer` (input) and `ImageBufferMut` (output), passed to filters during `Apply()` operations.

**IImageProcessor Interface**: Defines contract for image processors that work with `IImageSource` and `IImageSink`.

## Code Quality & Compiler Warnings

The project enforces strict compiler warning standards. All warnings are treated as errors for the project's own code (`-Wall`, `-Wextra`, `-Werror` configured in `.bazelrc`).

For parameters that are part of interface contracts but not used in specific implementations, the `[[maybe_unused]]` attribute is used to maintain interface compatibility while clearly indicating intentional non-use.

All code in `cpp_accelerator/` compiles without warnings when `-Werror` is enabled.

## C API Reference

The library exposes a C API through `processor_api.h` for language-agnostic integration. All data exchange uses Protocol Buffer serialization. The shared library build target (`libcuda_processor.so`) has been removed; the C API header is retained for the `processor_engine` wrapper used by the gRPC client.

**API Version**: The C API version is defined as `PROCESSOR_API_VERSION "2.1.0"` in `processor_api.h`. This is separate from the library version (4.0.2) and indicates the C interface contract.

## Adding New Filters

1. **Infrastructure**: Implement CPU and CUDA filter classes in `infrastructure/cpu/` and `infrastructure/cuda/`
2. **Application**: Filters are automatically usable via `FilterPipeline`
3. **Ports**: Update adapters if new parameters are required

The FilterPipeline automatically handles filter composition and execution order.

## Testing

Run all tests:
```bash
bazel test //src/cpp_accelerator/...
```

Run specific tests:
```bash
bazel test //src/cpp_accelerator/core:logger_test
bazel test //src/cpp_accelerator/core:result_test
bazel test //src/cpp_accelerator/application/pipeline:filter_pipeline_test
bazel test //src/cpp_accelerator/application/commands:commands_test
bazel test //src/cpp_accelerator/infrastructure/filters:blur_equivalence_test
bazel test //src/cpp_accelerator/infrastructure/cuda:grayscale_filter_test
bazel test //src/cpp_accelerator/infrastructure/cuda:blur_processor_test
bazel test //src/cpp_accelerator/infrastructure/cpu:grayscale_filter_test
bazel test //src/cpp_accelerator/infrastructure/cpu:blur_filter_test
bazel test //src/cpp_accelerator/infrastructure/image:image_loader_test
bazel test //src/cpp_accelerator/infrastructure/image:image_writer_test
bazel test //src/cpp_accelerator/infrastructure/config:config_manager_test
bazel test //src/cpp_accelerator/ports/shared_lib:blur_e2e_test
bazel test //src/cpp_accelerator/ports/grpc:data_channel_framing_test
```

## Building

Build accelerator control client:
```bash
bazel build //src/cpp_accelerator/ports/grpc:accelerator_control_client
```

Build all:
```bash
bazel build //src/cpp_accelerator/...
```

Refresh compile headers:
```bash
bazel run @hedron_compile_commands//:refresh_all
```

## Version Compatibility

The library uses semantic versioning:
- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible
