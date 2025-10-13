# Changelog

Completed features extracted from git commit history, organized by category.

## October 2025

### Native HTTPS Support (Oct 12, 2025)
- [x] Remove Caddy reverse proxy dependency
- [x] Implement native dual HTTP/HTTPS servers in Go using http.ListenAndServeTLS()
- [x] Add configurable HTTP and HTTPS ports with TLS settings
- [x] Centralize configuration to config/ directory
- [x] Simplify development workflow by removing proxy layer

### Connect-RPC Migration (Oct 12, 2025)
- [x] Migrate from stdlib HTTP to Connect-RPC
- [x] Add ImageProcessorService with ProcessImage and StreamProcessVideo RPCs
- [x] Implement Connect-RPC handlers in webserver/internal/interfaces/connectrpc
- [x] Refactor main.go to clean App structure with Run() method
- [x] Setup buf for proto code generation with Docker
- [x] Update Go to 1.24 in Bazel configuration
- [x] Remove old HTTP handlers and WebSocket implementation
- [x] Add HTTP annotations for REST-friendly endpoints

### Core Architecture (Oct 8, 2025)
- [x] Implement CUDA image processing with clean architecture
- [x] Reorganize into clean architecture layers with proper DI
- [x] Add final specifier to derived classes and improve code clarity
- [x] Integrate GoogleTest and GMock with Bazel
- [x] Reorganize C++ code into cpp_accelerator directory

### Testing Infrastructure (Oct 8, 2025)
- [x] Add unit tests for core logger
- [x] Add unit tests for config manager
- [x] Add unit tests for image adapters
- [x] Add unit tests for application commands

### Build System (Oct 9, 2025)
- [x] Integrate Golang with Bazel build system
- [x] Add Docker support with Bazel OCI rules and simplify Go build files

### Backend & Integration (Oct 9, 2025)
- [x] Implement web server with image filter UI and WebSocket support
- [x] Implement CGO integration with CUDA and real-time webcam streaming
- [x] Optimize video streaming performance and add resolution selector

### Image Processing (Oct 10, 2025)
- [x] Add multi-filter support with GPU/CPU selection
- [x] Implement multiple grayscale algorithms (BT.601, BT.709, Average, Lightness, Luminosity)

### HTTPS & Security (Oct 10, 2025)
- [x] Add HTTPS support with Caddy reverse proxy for local development
- [x] Configure Caddy reverse proxy with port 8443 for local HTTPS

### Frontend Development (Oct 10-11, 2025)
- [x] Enable dev mode with hot reload in start-dev.sh script
- [x] Redesign UI to dashboard layout with fixed sidebar and stats footer
- [x] Adjust dashboard UI spacing and filter auto-expand behavior
- [x] Modularize frontend architecture with toast notifications and auto-versioning
- [x] Simplify shell scripts to minimal professional style
- [x] Remove verbose comments and emojis from codebase

### Web Components Migration (Oct 11, 2025)
- [x] Setup Lit from CDN and migrate Camera to web component
- [x] Migrate camera component to Lit + TypeScript
- [x] Migrate Toast to Lit web component with TypeScript
- [x] Migrate Stats to Lit web component with TypeScript
- [x] Migrate Filters to Lit web component with TypeScript
- [x] Complete migration to Lit web components and TypeScript
- [x] Migrate frontend to Lit web components with Vite bundler
- [x] Split filter-panel into separate files

### Docker & Deployment (Oct 11, 2025)
- [x] Add Docker deployment with NVIDIA GPU support and Traefik HTTPS
- [x] Create production-ready Docker multi-stage build
- [x] Configure Traefik for HTTPS termination
- [x] Implement full GPU passthrough to container

### Documentation (Oct 10, 2025)
- [x] Add project README with setup and usage instructions
- [x] Add application screenshot

## Technical Achievements

### Architecture Patterns
- Clean Architecture with Domain/Application/Infrastructure layers
- Dependency Injection for testability
- Command pattern for filter pipeline
- Interface-based design for processors

### Technology Stack
- **Backend**: Go + CGO bridge to C++/CUDA
- **Build**: Bazel with C++23 support
- **Frontend**: Lit Web Components + TypeScript + Vite
- **Testing**: GoogleTest/GMock for C++
- **Containerization**: Docker with NVIDIA Container Toolkit
- **Reverse Proxy**: Caddy (dev) + Traefik (production)

### Performance
- Real-time video processing at ~150 FPS (GPU) vs ~25 FPS (CPU) at 720p
- WebSocket-based streaming with base64 PNG transport
- Frame-by-frame processing with stats tracking

## Next Milestones

See individual backlog files for upcoming features:
- [kernels.md](./docs/backlog/kernels.md) - More image processing filters
- [video-streaming.md](./docs/backlog/video-streaming.md) - Optimized video transport
- [neural-networks.md](./docs/backlog/neural-networks.md) - CUDA neural networks
- [infrastructure.md](./docs/backlog/infrastructure.md) - Microservices and observability

