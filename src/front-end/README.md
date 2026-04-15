# CUDA Image Processor - Frontend

TypeScript frontend application built with **Lit Web Components** and **React**, implementing Clean Architecture principles for maintainable and testable code.

## Overview

The frontend is a **multi-page application (MPA)** that provides real-time image and video processing through a web interface. It contains two dashboards:

1. **React Dashboard** (`/`) - Full React 19 application with components, hooks, and context providers
2. **Lit Dashboard** (`/lit`) - Original Lit Web Components application

Both communicate with the Go backend via Connect-RPC (gRPC-Web) for service calls and WebRTC for real-time frame streaming.

**Key Features:**
- Real-time webcam processing with GPU/CPU filter selection
- Static image and video file upload and processing
- Dynamic filter discovery from backend capabilities
- Drag-and-drop filter reordering
- Real-time performance metrics (FPS, processing time)
- Feature flag integration for gradual rollouts
- Comprehensive observability with OpenTelemetry

## Architecture

The frontend follows Clean Architecture principles with clear separation between domain interfaces, application services, infrastructure implementations, and UI components.

```mermaid
graph TB
    subgraph "UI Components Layer"
        AppRoot[app-root]
        FilterPanel[filter-panel]
        VideoGrid[video-grid]
        CameraPreview[camera-preview]
        StatsPanel[stats-panel]
        SourceDrawer[source-drawer]
        ToolsDropdown[tools-dropdown]
    end
    
    subgraph "Application Services Layer"
        ConfigService[ConfigService]
        ProcessorCaps[ProcessorCapabilitiesService]
        UIService[UIService]
    end
    
    subgraph "Infrastructure Layer"
        subgraph "Transport"
            WebRTCTransport[WebRTCFrameTransport]
        end
        
        subgraph "Data Services"
            VideoService[VideoService]
            FileService[FileService]
            InputSourceService[InputSourceService]
        end
        
        subgraph "External Services"
            FeatureFlags[FeatureFlagsService]
            SystemInfo[SystemInfoService]
            ToolsService[ToolsService]
        end
        
        subgraph "Observability"
            TelemetryService[TelemetryService]
            Logger[Logger]
        end
    end
    
    subgraph "Domain Layer"
        Interfaces[Domain Interfaces]
        ValueObjects[Value Objects]
    end
    
    subgraph "Backend"
        GoServer[Go Web Server]
        GRPCServer[gRPC Server]
    end
    
    AppRoot --> FilterPanel
    AppRoot --> VideoGrid
    AppRoot --> CameraPreview
    AppRoot --> StatsPanel
    AppRoot --> SourceDrawer
    
    FilterPanel --> ProcessorCaps
    CameraPreview --> WebRTCTransport
    VideoGrid --> VideoService
    
    ProcessorCaps --> GRPCServer
    WebRTCTransport --> GRPCServer
    
    VideoService --> GoServer
    FileService --> GoServer
    InputSourceService --> GoServer
    
    ConfigService --> GoServer
    FeatureFlags --> GoServer
    SystemInfo --> GoServer
    ToolsService --> GoServer
    
    UIService --> WebRTCTransport
    UIService --> ProcessorCaps
    
    ConfigService --> Interfaces
    ProcessorCaps --> Interfaces
    WebRTCTransport --> Interfaces
    VideoService --> Interfaces
    FileService --> Interfaces
    
    TelemetryService --> Logger
    Logger --> GoServer
```

## Component Structure

### Core Application Components

**`app-root`** (`components/app/app-root.ts`):
- Main application container component
- Manages global state and component lifecycle
- Coordinates between UI components and services
- Handles application initialization

**`filter-panel`** (`components/app/filter-panel.ts`):
- Displays available filters from backend capabilities
- Allows drag-and-drop filter reordering
- Provides parameter controls (select, range, number, checkbox, text)
- Updates filter configuration in real-time

**`video-grid`** (`components/video/video-grid.ts`):
- Displays video sources in a grid layout
- Handles video selection and playback
- Manages video source cards

**`camera-preview`** (`components/video/camera-preview.ts`):
- Captures frames from webcam using MediaDevices API
- Sends frames to backend via transport layer
- Displays processed frames in real-time
- Shows connection status and performance metrics

**`stats-panel`** (`components/app/stats-panel.ts`):
- Displays real-time processing statistics
- Shows FPS, processing time, frame count
- Toggleable visibility with state persistence

**`source-drawer`** (`components/app/source-drawer.ts`):
- Side drawer for selecting input sources
- Tabs for Images and Videos
- Integrates with ImageSelectorModal and VideoSelector

### Supporting Components

**`toast-container`** (`components/app/toast-container.ts`):
- Global toast notification system
- Success, error, warning, and info messages
- Auto-dismiss with configurable duration

**`app-tour`** (`components/app/app-tour.ts`):
- Guided tour for first-time users
- Highlights key features and UI elements
- Uses Shepherd.js for tour management

**`connection-status-card`** (`components/app/connection-status-card.ts`):
- Displays current connection status
- Shows WebRTC connection state and quality
- Connection quality indicators

**`tools-dropdown`** (`components/ui/tools-dropdown.ts`):
- Dynamic tools menu based on configuration
- Links to observability tools (Jaeger, Grafana)
- Feature flag management (Flipt)
- Test reports and coverage

## Service Architecture

### Application Services (`application/services/`)

**`config-service.ts`**:
- Manages stream configuration from backend
- Handles log level and console logging settings
- Provides configuration to other services

**`processor-capabilities-service.ts`**:
- Fetches available filters and their parameters
- Maps generic filter definitions to UI types
- Notifies listeners when capabilities change
- Caches filter definitions for performance

**`ui-service.ts`**:
- Coordinates UI interactions
- Manages stats, camera, and filter state
- Bridges components and transport layer

### Infrastructure Services

#### Transport Layer (`infrastructure/transport/`)

**`webrtc-frame-transport.ts`**:
- WebRTC implementation for real-time frame streaming
- Peer-to-peer communication with the gRPC server
- Uses WebRTC data channels for low-latency frame transmission
- Handles WebRTC signaling through Connect-RPC
- Connection lifecycle management and reconnection handling

#### Data Services (`infrastructure/data/`)

**`video-service.ts`**:
- Video file management (list, upload)
- Video metadata handling
- Preview image generation

**`file-service.ts`**:
- Static image file management
- Image upload and listing
- File validation

**`input-source-service.ts`**:
- Manages available input sources
- Coordinates between images, videos, and webcam
- Source selection and switching

#### External Services (`infrastructure/external/`)

**`feature-flags-service.ts`**:
- Fetches and caches feature flags from Flipt
- Evaluates boolean and variant flags
- Supports gradual rollouts

**`system-info-service.ts`**:
- Retrieves system information from backend
- Version information and build details
- Backend capabilities

**`tools-service.ts`**:
- Dynamic tools configuration
- Observability tool links
- Test report access

#### Observability (`infrastructure/observability/`)

**`telemetry-service.ts`**:
- OpenTelemetry integration
- Distributed tracing support
- Span creation and management

**`otel-logger.ts`**:
- Structured logging with OpenTelemetry
- Log level management
- Console and remote logging

## Domain Layer

### Interfaces (`domain/interfaces/`)

Domain interfaces define contracts without implementation details:

- **`IFrameTransportService`**: Frame transmission interface (WebRTC implementation)
- **`IConfigService`**: Configuration management
- **`IProcessorCapabilitiesService`**: Filter capabilities
- **`IVideoService`**: Video operations
- **`IFileService`**: File operations
- **`IInputSourceService`**: Input source management
- **`ITelemetryService`**: Observability
- **`ILogger`**: Logging interface

### Value Objects (`domain/value-objects/`)

Type-safe domain models:

- **`ImageData`**: Image data with dimensions and format
- **`FilterData`**: Filter configuration and parameters
- **`AcceleratorConfig`**: GPU/CPU accelerator selection
- **`GrayscaleAlgorithm`**: Grayscale algorithm types
- **`ConnectionStatus`**: WebRTC connection state and quality metrics
- **`WebRTCSession`**: WebRTC session information

## Dependency Injection

**`application/di/Container.ts`**:
- Centralized dependency injection container
- Singleton pattern for service instances
- Factory methods for component-specific services
- Provides type-safe service access

**Service Resolution:**
- Application services: Singleton instances
- Transport service: WebRTCFrameTransportService with component dependencies
- Infrastructure services: Singleton instances with lazy initialization

## Transport Selection

The frontend uses **WebRTC** as the primary transport for real-time frame streaming:

- **WebRTC**: Peer-to-peer low-latency streaming via WebRTC data channels
- Signaling is handled through Connect-RPC WebRTC services
- Components use the `IFrameTransportService` interface, implemented by `WebRTCFrameTransportService`

The transport provides:
- Direct browser-to-gRTC server communication
- Low-latency frame transmission for real-time processing
- Automatic connection management and reconnection

## Development

### Quick Start

From project root:
```bash
./scripts/dev/start.sh --build  # First time or after code changes
./scripts/dev/start.sh           # Subsequent runs (hot reload)
```

**Access:**
- **React Dashboard**: https://localhost:8443 (or http://localhost:3000 for Vite dev server)
- **Lit Dashboard**: https://localhost:8443/lit (or http://localhost:3000/lit for Vite dev server)

The backend Go server runs on port 8443, while the Vite dev server runs on port 3000 during development.

### Manual Development

```bash
cd src/front-end
npm install
npm run dev  # Vite dev server with hot reload on port 3000
```

### Build

```bash
npm run build  # Production build for both React and Lit
```

The build output is embedded in the Go server binary as static assets. The production deployment uses Nginx to serve the pre-built static files.

## Testing

### Unit Tests

```bash
npm run test  # Vitest unit tests
npm run test:ui  # Vitest UI mode
npm run test:coverage  # Generate coverage report
```

### E2E Tests

```bash
npm run test:e2e  # Playwright E2E tests
npm run test:e2e:ui  # Playwright UI mode
npm run test:e2e:dev  # Development mode
```

## Tech Stack

- **LitElement**: Native web components framework
- **React 19**: Modern React application with hooks and context
- **TypeScript**: Type-safe JavaScript
- **Vite**: Build tool and dev server with MPA support
- **Vitest**: Unit testing framework
- **Playwright**: E2E testing
- **Connect-RPC**: gRPC-Web client library for service calls and WebRTC signaling
- **WebRTC**: Real-time peer-to-peer frame streaming
- **OpenTelemetry**: Distributed tracing
- **Shepherd.js**: Guided tour library

## Directory Structure

```
front-end/
├── src/
│   ├── lit/                 # Lit Web Components application
│   │   ├── components/      # LitElement components
│   │   │   ├── app/        # Core application components
│   │   │   ├── video/      # Video-related components
│   │   │   ├── image/      # Image-related components
│   │   │   ├── flags/      # Feature flag components
│   │   │   └── ui/         # UI utility components
│   │   ├── application/    # Application layer (services, DI)
│   │   ├── infrastructure/ # Infrastructure layer
│   │   ├── domain/         # Domain layer
│   │   └── main.ts         # Lit entry point
│   ├── react/               # React application
│   │   ├── components/     # React components
│   │   │   ├── app/        # Core app components
│   │   │   ├── filters/    # Filter components
│   │   │   ├── health/     # Health monitoring
│   │   │   ├── image/      # Image processing
│   │   │   ├── video/      # Video components
│   │   │   ├── files/      # File management
│   │   │   ├── settings/   # Settings panel
│   │   │   └── sidebar/    # Sidebar components
│   │   ├── hooks/          # React hooks
│   │   ├── context/        # React context providers
│   │   ├── providers/      # Service providers
│   │   ├── infrastructure/ # React infrastructure
│   │   └── main.tsx        # React entry point
│   ├── services/            # Shared services
│   ├── gen/                 # Generated protobuf code
│   └── domain/              # Shared domain layer
├── public/                  # Static assets
│   └── static/             # CSS, images, fonts
├── tests/
│   └── e2e/                # E2E tests (Playwright)
├── index.html               # Lit dashboard entry
├── react.html               # React dashboard entry
├── vite.config.ts           # Vite MPA configuration
├── Dockerfile               # Production build with Nginx
└── package.json
```

## Design Principles

1. **Clean Architecture**: Clear separation between domain, application, and infrastructure
2. **Dependency Inversion**: Components depend on interfaces, not implementations
3. **Single Responsibility**: Each component and service has one clear purpose
4. **Interface Segregation**: Small, focused interfaces
5. **Composition over Inheritance**: Services aggregate functionality
6. **Type Safety**: TypeScript for compile-time error detection

## See Also

- [Main README](../../README.md) - Project overview
- [Go API README](../go_api/README.md) - Backend architecture
- [Testing Documentation](../../docs/testing-and-coverage.md) - Test execution guide
