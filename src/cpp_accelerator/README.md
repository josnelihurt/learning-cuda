# CUDA Accelerator Library

High-performance image processing library implementing Clean Architecture principles with multi-backend GPU acceleration (CUDA, OpenCL) and CPU fallback support.

## Library Description

The CUDA Accelerator Library provides a production-grade image processing framework with GPU-accelerated filters. The architecture uses a **pluggable filter factory** pattern: `ProcessorEngine` is decoupled from concrete filter implementations via `IFilterFactory`, with one factory registered per accelerator backend (CUDA, CPU, OpenCL). New backends are added by implementing `IFilterFactory` and registering it at startup — no engine changes required.

**Version**: See `VERSION` file (currently 4.0.9)

**Features**:
- Multi-backend GPU acceleration: **CUDA**, **OpenCL**, and CPU fallback
- Pluggable `IFilterFactory` / `FilterFactoryRegistry` for extensibility
- **Accelerator Control Client** with mTLS outbound connections to Go cloud server
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

    subgraph "Adapters / Compute"
        CudaFactory[CUDA FilterFactory]
        CpuFactory[CPU FilterFactory]
        OpenCLFactory[OpenCL FilterFactory]
        VulkanFactory[Vulkan FilterFactory]
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
    ProcessorEngine -->|FilterFactoryRegistry| CudaFactory
    ProcessorEngine -->|FilterFactoryRegistry| CpuFactory
    ProcessorEngine -->|FilterFactoryRegistry| OpenCLFactory
    ProcessorEngine -->|FilterFactoryRegistry| VulkanFactory
    CudaFactory --> IFilter
    CpuFactory --> IFilter
    OpenCLFactory --> IFilter
    VulkanFactory --> IFilter
    CudaFactory --> YOLODetector
    YOLODetector --> ModelManager
    FilterPipeline --> ImageBuffer
```

### Layer Structure

```mermaid
graph TB
    subgraph "Go Cloud Server"
        GoServer[Go Server<br/>AcceleratorControlService]
    end

    subgraph "C++ Ports (interfaces)"
        IControlPort[IControlPort]
        IMediaSession[IMediaSession]
    end

    subgraph "C++ Adapters"
        AccelClient[adapters/grpc_control<br/>AcceleratorControlClient]
        Provider[ProcessorEngineAdapter]
        WebRTCPort[adapters/webrtc<br/>WebRTCManager]
        DataChannelFraming[DataChannelFraming]
        LiveVideoProcessor[LiveVideoProcessor]
    end

    subgraph "C++ Application Layer"
        ProcessorEngine[application/engine<br/>ProcessorEngine]
        FactoryRegistry[FilterFactoryRegistry]
        ServerInfo[server_info<br/>ServerInfoProvider]
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

    subgraph "C++ Adapters / Compute"
        CudaFactory[compute/cuda<br/>CudaFilterFactory]
        CpuFactory[compute/cpu<br/>CpuFilterFactory]
        OpenCLFactory[compute/opencl<br/>OpenCLFilterFactory]
        VulkanFactory[compute/vulkan<br/>VulkanFilterFactory]
        TRTInference[cuda/tensorrt<br/>YOLO + TensorRT]
        ImageIO[image_io<br/>Image I/O]
        ConfigManager[config<br/>ConfigManager]
    end

    subgraph "C++ Core Layer"
        Logger[Logger]
        Telemetry[Telemetry]
        Result[Result Type]
    end

    GoServer -->|mTLS bidi| AccelClient
    AccelClient -.->|implements| IControlPort
    WebRTCPort -.->|implements| IMediaSession
    AccelClient --> Provider
    Provider --> ProcessorEngine
    AccelClient --> WebRTCPort
    WebRTCPort --> ServerInfo

    ProcessorEngine --> FilterPipeline
    FilterPipeline --> BufferPool
    FilterPipeline --> IFilter
    ProcessorEngine --> FactoryRegistry
    FactoryRegistry --> CudaFactory
    FactoryRegistry --> CpuFactory
    FactoryRegistry --> OpenCLFactory
    FactoryRegistry --> VulkanFactory
    CudaFactory --> IFilter
    CpuFactory --> IFilter
    OpenCLFactory --> IFilter
    VulkanFactory --> IFilter
    CudaFactory --> TRTInference
    CommandFactory --> ConfigManager
    ProcessorEngine --> Logger
    ProcessorEngine --> Telemetry
```

### Initialization Sequence

```mermaid
sequenceDiagram
    participant Main as cmd/accelerator_control_client/main.cpp
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
    participant Info as ServerInfoProvider

    Peer->>WRTC: ControlRequest(kListFilters) on "control" channel
    WRTC->>Info: PopulateListFiltersResponse()
    Info-->>WRTC: filter definitions
    WRTC->>Peer: ControlResponse(ListFiltersResponse)

    Peer->>WRTC: ControlRequest(kGetVersion) on "control" channel
    WRTC->>Info: PopulateVersionResponse()
    Info-->>WRTC: version + build info
    WRTC->>Peer: ControlResponse(GetVersionInfoResponse)
```

## Directory Structure

Hexagonal architecture: `ports/` holds abstract interfaces only; `adapters/` holds all concrete implementations.

```
cpp_accelerator/
├── cmd/
│   ├── accelerator_control_client/
│   │   ├── main.cpp            # Binary entry point
│   │   └── BUILD
│   ├── spike_multi_gpu_backend/  # Experimental multi-backend abstraction spike
│   └── [hello-world examples]    # See "Hello World Examples" section below
├── core/                       # Cross-cutting utilities (no deps on other layers)
│   ├── logger.h/cpp
│   ├── telemetry.h/cpp
│   ├── otel_log_sink.h/cpp
│   ├── signal_handler.h/cpp
│   ├── result.h
│   └── version.h               # Build-time version + git hash (generated)
├── domain/                     # Pure domain — no infrastructure deps
│   ├── interfaces/
│   │   ├── filters/i_filter.h
│   │   ├── processors/i_image_processor.h
│   │   ├── i_yolo_detector.h   # YOLO detector interface (extends IFilter)
│   │   ├── image_buffer.h, image_sink.h, image_source.h, i_pixel_getter.h
│   │   └── grayscale_algorithm.h
│   └── models/
│       └── detection.h         # Detection result struct
├── application/                # Use cases — orchestrates domain + adapters
│   ├── engine/
│   │   ├── processor_engine.h/cpp   # Main orchestrator: filter dispatch + YOLO cache
│   │   ├── i_filter_factory.h       # IFilterFactory — one per accelerator backend
│   │   ├── filter_descriptor.h      # FilterDescriptor, FilterCreationParams, BlurBorderMode
│   │   ├── filter_factory_registry.h/cpp  # Registry: AcceleratorType → IFilterFactory
│   │   └── BUILD
│   ├── pipeline/
│   │   ├── filter_pipeline.h/cpp
│   │   └── buffer_pool.h/cpp
│   ├── server_info/
│   │   ├── i_server_info_provider.h     # Interface for version/filter queries
│   │   ├── server_info_provider.h/cpp   # Implementation (reads VERSION, queries engine caps)
│   │   ├── filter_parameter_mapping.h/cpp  # Engine FilterParameter → wire GenericFilterParameter
│   │   └── accelerator_label.h/cpp         # AcceleratorType → human-readable label
│   └── commands/               # CommandFactory (not wired into main pipeline)
├── ports/                      # Abstract port interfaces ONLY
│   ├── control/
│   │   └── i_control_port.h    # IControlPort — Run/Stop interface
│   └── media/
│       └── i_media_session.h   # IMediaSession — WebRTC session lifecycle
├── adapters/                   # Concrete implementations of ports + domain
│   ├── grpc_control/           # Outbound mTLS gRPC control client
│   │   ├── accelerator_control_client.h/cpp  # implements IControlPort
│   │   ├── processor_engine_adapter.h/cpp    # bridges engine → ProcessorEngineProvider
│   │   └── processor_engine_provider.h       # service provider interface
│   ├── webrtc/                 # WebRTC media path
│   │   ├── webrtc_manager.h/cpp              # implements IMediaSession
│   │   ├── live_video_processor.h/cpp        # H.264 decode → ProcessorEngine → encode
│   │   ├── data_channel_framing.h/cpp        # chunked binary framing + SendFramed over SCTP
│   │   ├── channel_labels.h                  # channel name constants, session prefixes
│   │   ├── session_routing.h/cpp             # IsGoVideoSession, ShouldRegisterSessionChannel
│   │   ├── protocol/             # WebRTC protocol helpers
│   │   │   ├── filter_resolver.h/cpp         # Generic filter → enum mapping
│   │   │   └── data_channel_envelope.h/cpp   # DataChannelRequest protobuf envelope parsing
│   │   └── sdp/                  # SDP utilities
│   │       └── sdp_utils.h/cpp               # Codec negotiation, extmap strip, ICE injection
│   ├── compute/
│   │   ├── cpu/                # CPU filter implementations
│   │   │   ├── grayscale_filter.h/cpp
│   │   │   ├── blur_filter.h/cpp
│   │   │   └── cpu_filter_factory.h/cpp    # IFilterFactory for CPU
│   │   ├── cuda/
│   │   │   ├── cuda_filter_factory.h/cpp   # IFilterFactory for CUDA
│   │   │   ├── kernels/        # Kernel launchers (.h) and CUDA source (.cu)
│   │   │   │   ├── grayscale_kernel.h/cu
│   │   │   │   ├── letterbox_kernel.h/cu
│   │   │   │   ├── blur_kernel.h           # Launcher header for blur variants
│   │   │   │   └── blur/       # Blur kernel implementations
│   │   │   │       ├── device_utils.cuh    # Shared border-clamp helpers
│   │   │   │       ├── non_separable.cu    # Naïve 2D convolution (reference)
│   │   │   │       ├── separable_basic.cu  # Two 1D passes (algorithmic win)
│   │   │   │       └── separable_tiled.cu  # Tiled + constant mem (production)
│   │   │   ├── filters/        # C++ wrappers: grayscale_filter, blur_filter
│   │   │   ├── memory/         # cuda_memory_pool (thread-local GPU alloc cache)
│   │   │   └── tensorrt/       # TensorRT/YOLO inference
│   │   │       ├── yolo_detector.h/cpp
│   │   │       ├── yolo_factory.h/trt.cpp
│   │   │       ├── model_manager.h/cpp
│   │   │       └── model_registry.h/cpp
│   │   ├── opencl/              # OpenCL filter implementations
│   │   │   ├── context/
│   │   │   │   └── context.h/cpp            # OpenCL platform/device initialization
│   │   │   ├── filters/
│   │   │   │   ├── grayscale_filter.h/cpp
│   │   │   │   └── blur_filter.h/cpp
│   │   │   ├── kernels/
│   │   │   │   ├── cl_grayscale.cl          # Grayscale OpenCL kernel source
│   │   │   │   └── cl_blur.cl               # Blur OpenCL kernel source
│   │   │   └── opencl_filter_factory.h/cpp  # IFilterFactory for OpenCL
│   │   ├── vulkan/              # Vulkan compute filter implementations
│   │   │   ├── context/
│   │   │   │   ├── context.h/cpp            # Vulkan instance/device initialization
│   │   │   │   └── compute_utils.h
│   │   │   ├── filters/
│   │   │   │   ├── grayscale_filter.h/cpp
│   │   │   │   └── blur_filter.h/cpp
│   │   │   ├── kernels/
│   │   │   │   ├── grayscale.comp           # Grayscale GLSL compute shader
│   │   │   │   └── blur.comp                # Blur GLSL compute shader
│   │   │   └── vulkan_filter_factory.h/cpp  # IFilterFactory for Vulkan
│   │   └── filters/            # Cross-backend equivalence tests
│   ├── image_io/               # Image file I/O (stb-based)
│   │   ├── image_loader.h/cpp
│   │   └── image_writer.h/cpp
│   └── config/                 # Configuration management
│       ├── config_manager.h/cpp
│       └── models/program_config.h
├── composition/                # Bazel-flag-driven platform wiring
│   └── platform/
│       ├── platform_support.h              # Common registration interface
│       ├── platform_support_cpu.cpp        # CPU-only build
│       ├── platform_support_cuda.cpp       # CUDA-only build
│       ├── platform_support_opencl.cpp     # OpenCL-only build
│       ├── platform_support_vulkan.cpp     # Vulkan-only build
│       ├── platform_support_full.cpp       # All backends
│       ├── platform_support_all.cpp        # Alias: all backends
│       ├── platform_support_cuda_vulkan.cpp
│       ├── platform_support_opencl_vulkan.cpp
│       ├── cpu/
│       │   └── cpu_platform.h/cpp          # CPU platform registration
│       ├── cuda/
│       │   └── cuda_platform.h/cpp         # CUDA platform registration
│       ├── opencl/
│       │   └── opencl_platform.h/cpp       # OpenCL platform registration
│       └── vulkan/
│           └── vulkan_platform.h/cpp       # Vulkan platform registration
├── docker-cuda-runtime/
├── yolo-model-gen/
├── Dockerfile.build
├── VERSION
└── lessons_learned.md
```

## UML Class Diagrams

Mermaid class diagrams for each architectural layer are generated automatically from source using [clang-uml](https://github.com/bkryza/clang-uml) and the `compile_commands.json` produced by Bazel.

| Diagram | Scope |
|---------|-------|
| [cpp_domain_layer](../../docs/uml/generated/cpp_domain_layer.mmd) | `jrb::domain` — interfaces & models |
| [cpp_application_layer](../../docs/uml/generated/cpp_application_layer.mmd) | `jrb::application` — engine, pipeline, factories |
| [cpp_ports](../../docs/uml/generated/cpp_ports.mmd) | `jrb::ports` — hexagonal port interfaces |
| [cpp_core](../../docs/uml/generated/cpp_core.mmd) | `jrb::core` — Logger, Telemetry, Result |
| [cpp_control_adapters](../../docs/uml/generated/cpp_control_adapters.mmd) | gRPC control + WebRTC adapters |
| [cpp_compute_cpu](../../docs/uml/generated/cpp_compute_cpu.mmd) | `jrb::adapters::compute::cpu` |
| [cpp_compute_cuda](../../docs/uml/generated/cpp_compute_cuda.mmd) | `jrb::adapters::compute::cuda` + TensorRT |
| [cpp_compute_opencl](../../docs/uml/generated/cpp_compute_opencl.mmd) | `jrb::adapters::compute::opencl` |
| [cpp_compute_vulkan](../../docs/uml/generated/cpp_compute_vulkan.mmd) | `jrb::adapters::compute::vulkan` |

**Regenerate diagrams** (after source changes):

```bash
# One-time install (Ubuntu 24.04)
scripts/dev/install-clang-uml.sh

# Generate / refresh
scripts/build/uml.sh
```

See [docs/uml/README.md](../../docs/uml/README.md) for full documentation, viewing options, and troubleshooting.

## Sub-folder Documentation

- **[adapters/compute/cuda/README.md](adapters/compute/cuda/README.md)** — Comprehensive CUDA tutorial covering kernel implementations, memory hierarchy, blur optimization variants, letterbox preprocessing, and TensorRT YOLO inference pipeline.

- **[application/engine/README.md](application/engine/README.md)** — Describes the Strategy Pattern dispatch used in `IFilterFactory::CreateFilter` and the `DispatchCreateFilter` shared helper.

## Hello World Examples

The `cmd/` folder contains several minimal examples for learning GPU programming concepts. These are standalone programs that demonstrate specific compute APIs without the full accelerator architecture:

- **[OpenCL Hello World](cmd/hello-world-opencl/README.md)** — Minimal OpenCL example adding two float vectors on the GPU. Demonstrates OpenCL platform/device initialization, embedded SPIR-V and OpenCL C source assets, and program compilation.

- **[Vulkan Compute Hello World](cmd/hello-world-vulkan/README.md)** — Minimal Vulkan compute shader example for vector addition. Demonstrates Vulkan instance/device setup, compute pipeline creation, and embedded SPIR-V shaders.

- **[Spike Multi-GPU Backend](cmd/spike_multi_gpu_backend/)** — Experimental spike exploring a unified backend abstraction for CUDA/OpenCL with a grayscale filter demo.

These examples are intentionally kept separate from the production codebase and serve as learning resources for understanding the underlying compute APIs.

## Design Principles

1. **Dependency Inversion**: Domain interfaces define contracts; infrastructure implements them
2. **Single Responsibility**: Each component has one clear purpose
3. **Open/Closed**: Extend via new implementations, not modification
4. **Liskov Substitution**: All filter implementations are interchangeable
5. **Interface Segregation**: Small, focused interfaces (IFilter, ImageBuffer)
6. **Separation of Concerns**: Clear boundaries between layers
7. **Filter Pipeline**: Composable filter architecture for chaining multiple filters
8. **Pluggable Backends**: `IFilterFactory` + `FilterFactoryRegistry` enable zero-change engine extensibility

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

- **WebRTCManager** (`adapters/webrtc/webrtc_manager.h/cpp`): Manages WebRTC peer connections, ICE candidate exchange, session lifecycle, and per-session CUDA memory pools
- **LiveVideoProcessor** (`adapters/webrtc/live_video_processor.h/cpp`): Real-time video frame processing pipeline — FFmpeg H.264 decode → RGB → ProcessorEngine → RGB → FFmpeg H.264 encode
- **DataChannelFraming** (`adapters/webrtc/data_channel_framing.h/cpp`): Binary chunking/reassembly protocol for large protobuf messages over SCTP data channels, plus `SendFramed()` helper
- **Channel labels & routing** (`adapters/webrtc/channel_labels.h`, `session_routing.h/cpp`): Channel name constants, Go-video session prefix detection, session channel registration logic
- **Protocol helpers** (`adapters/webrtc/protocol/`): Filter parameter resolution (`filter_resolver`), `DataChannelRequest` protobuf envelope parsing (`data_channel_envelope`)
- **SDP utilities** (`adapters/webrtc/sdp/sdp_utils.h/cpp`): H.264 codec negotiation, RTP header extension stripping, manual ICE candidate injection, SDP answer waiting

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

The library includes YOLO object detection via TensorRT for GPU-accelerated inference. See the [CUDA compute adapter README](adapters/compute/cuda/README.md) for detailed documentation on the inference pipeline, letterbox preprocessing, and TensorRT engine lifecycle.

**Components**:

- **IYoloDetector** (`domain/interfaces/i_yolo_detector.h`): Detector interface extending `IFilter`
- **YOLODetector** (`adapters/compute/cuda/tensorrt/yolo_detector.h/cpp`): TensorRT-based inference engine with NMS post-processing
- **YoloFactory** (`adapters/compute/cuda/tensorrt/yolo_factory.h/trt.cpp`): Factory for creating detector instances
- **ModelManager** (`adapters/compute/cuda/tensorrt/model_manager.h/cpp`): Model loading & session management
- **ModelRegistry** (`adapters/compute/cuda/tensorrt/model_registry.h/cpp`): Model path resolution
- **LetterboxKernel** (`adapters/compute/cuda/kernels/letterbox_kernel.cu/h`): GPU-accelerated resize + pad + NCHW conversion

### Command Pattern

The command pattern infrastructure (`application/commands/`) is maintained for potential future use. All processing is currently handled directly by `FilterPipeline` which orchestrates filter chains without the command pattern abstraction layer.

### Filter Factory Pattern

The engine uses a **pluggable factory pattern** to decouple from concrete filter implementations:

- **`IFilterFactory`** (`application/engine/i_filter_factory.h`): Interface with three methods — `GetAcceleratorType()`, `GetFilterDescriptors()`, `CreateFilter()`
- **`FilterFactoryRegistry`** (`application/engine/filter_factory_registry.h`): Maps `AcceleratorType` → `IFilterFactory`. One factory per type; re-registration replaces.
- **`FilterDescriptor`** (`application/engine/filter_descriptor.h`): Declarative filter metadata (id, name, parameters with validation rules). Each factory returns its supported filters.
- **`FilterCreationParams`**: Runtime parameter struct passed to `CreateFilter()` — factories read what they need.

Registered factories: `CudaFilterFactory`, `CpuFilterFactory`, `OpenCLFilterFactory`, `VulkanFilterFactory`.

`GetCapabilities()` accepts an optional `requested_accelerator` — when specified, only that backend's filter descriptors are returned. When unspecified, all factories contribute (backward compatible).

### Processor Engine

The `ProcessorEngine` is the core orchestration component that coordinates image processing operations. It uses a `FilterFactoryRegistry` to look up the appropriate `IFilterFactory` for the requested accelerator type, then delegates filter creation to that factory. This decouples the engine from concrete filter implementations.

**Responsibilities**: Initialization and telemetry setup, filter orchestration via `FilterFactoryRegistry` → `IFilterFactory` → `FilterPipeline`, algorithm selection from protocol buffer enums, and response building.

**Integration Points**: Used by `adapters/grpc_control` via `ProcessorEngineAdapter` for the gRPC service, and by `adapters/webrtc` for WebRTC data channel and live video processing.

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

The library previously exposed a C API through `processor_api.h`. The shared library build target (`libcuda_processor.so`) has been removed. The `processor_engine` wrapper in the application layer is now used directly by the gRPC and WebRTC adapters.

**API Version**: The C API version was defined as `PROCESSOR_API_VERSION "2.1.0"`. This is separate from the library version (4.0.6).

## Adding New Filters or Backends

### Adding a new filter to an existing backend

1. **Adapters**: Implement the filter class in the appropriate `adapters/compute/<backend>/` directory
2. **Factory**: Add the filter to the backend's `IFilterFactory::CreateFilter()` switch and add a `FilterDescriptor` to `GetFilterDescriptors()`
3. **Protocol**: Add parameter parsing in `adapters/webrtc/protocol/filter_resolver` if new generic filter parameters are needed

### Adding a new accelerator backend

1. **Adapters**: Create `adapters/compute/<backend>/` with filter implementations and an `<backend>_filter_factory.h/cpp` implementing `IFilterFactory`
2. **Engine**: Register the factory in `ProcessorEngine` constructor: `factory_registry_.Register(std::make_unique<NewBackendFilterFactory>())`
3. **Protocol buffer**: Add the new `AcceleratorType` enum value in `image_processor_service.proto`

The `FilterFactoryRegistry` handles the rest — no engine logic changes needed.

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
bazel test //src/cpp_accelerator/adapters/compute/filters:blur_equivalence_test
bazel test //src/cpp_accelerator/adapters/compute/cuda/filters:grayscale_filter_test
bazel test //src/cpp_accelerator/adapters/compute/cuda/filters:blur_filter_test
bazel test //src/cpp_accelerator/adapters/compute/cpu:grayscale_filter_test
bazel test //src/cpp_accelerator/adapters/compute/cpu:blur_filter_test
bazel test //src/cpp_accelerator/adapters/image_io:image_loader_test
bazel test //src/cpp_accelerator/adapters/image_io:image_writer_test
bazel test //src/cpp_accelerator/adapters/config:config_manager_test
bazel test //src/cpp_accelerator/adapters/webrtc:data_channel_framing_test
```

## Building

Build accelerator control client:
```bash
bazel build //src/cpp_accelerator/cmd/accelerator_control_client
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
