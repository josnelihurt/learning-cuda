# Changelog

Completed features extracted from git commit history, organized by category.

## October 2025

### Code Quality & Testing Infrastructure (Oct 18, 2025)
- [x] Configure comprehensive linting for all languages (C++, Go, TypeScript)
- [x] Add clang-tidy configuration with 50+ checks for C++ code quality
- [x] Add golangci-lint with 20+ linters for Go code analysis
- [x] Add ESLint with TypeScript, Lit, and Web Components support for frontend
- [x] Integrate Prettier for consistent code formatting across frontend
- [x] Create unified run-linters.sh script with --fix option for auto-corrections
- [x] Add Docker services for linting (lint-frontend, lint-golang, lint-cpp)
- [x] Configure Vitest for frontend unit testing with coverage reporting
- [x] Create run-coverage.sh script for comprehensive test coverage collection
- [x] Add coverage-report-viewer Docker service on port 5052
- [x] Implement Go unit tests for ProcessImageUseCase with 12 test cases (211 lines)
- [x] Implement frontend unit tests for ToastContainer with 8 test suites (199 lines)
- [x] Configure coverage thresholds (80% for lines, functions, branches, statements)
- [x] Add coverage reports for frontend (HTML, JSON, LCOV), Golang (HTML), and C++ (LCOV)
- [x] Update README with Testing & Code Quality section
- [x] Enhanced .clang-format configuration with 68 formatting rules
- [x] All linters passing with zero errors across codebase
- [x] Total test coverage infrastructure across 3 languages

### End-to-End Testing with Playwright (Oct 17, 2025)
- [x] Integrate Playwright testing framework with TypeScript
- [x] Add 9 comprehensive E2E test suites (954 lines of tests)
- [x] Implement drawer-functionality tests (82 lines, 4 tests)
- [x] Implement filter-configuration tests (86 lines, 5 tests)
- [x] Implement filter-toggle tests (79 lines, 4 tests)
- [x] Implement multi-source-management tests (85 lines, 4 tests)
- [x] Implement panel-synchronization tests (97 lines, 5 tests)
- [x] Implement resolution-control tests (71 lines, 3 tests)
- [x] Implement source-removal tests (55 lines, 2 tests)
- [x] Implement ui-validation tests (162 lines, 8 tests)
- [x] Implement websocket-management tests (55 lines, 2 tests)
- [x] Create test-helpers utility with 182 lines of reusable functions
- [x] Add data-testid attributes to all interactive frontend components
- [x] Configure Playwright for Chrome, Firefox, and WebKit testing
- [x] Create run-e2e-tests.sh script for automated test execution
- [x] Integrate E2E tests with Docker Compose for CI/CD
- [x] Add Playwright service to docker-compose.dev.yml
- [x] Configure test artifacts and reports output
- [x] Add npm scripts for running tests (headed/headless/debug/UI modes)
- [x] Update .gitignore for Playwright artifacts
- [x] All tests passing with cross-browser validation

### Dynamic Tools Configuration (Oct 17, 2025)
- [x] Create Tool and ToolCategory proto messages with support for URLs and actions
- [x] Add GetAvailableTools RPC endpoint to ConfigService
- [x] Implement tools configuration in config.yaml with dev/prod URL mappings
- [x] Add environment detection via CUDA_PROCESSOR_ENVIRONMENT env variable
- [x] Create ToolsService for dynamic tool discovery from backend
- [x] Refactor tools-dropdown component to render tools dynamically
- [x] Remove hardcoded tool URLs and port-based environment detection from frontend
- [x] Download and serve favicons locally for improved performance
- [x] Add 6 BDD scenarios for tools configuration validation
- [x] Support both URL tools (external links) and action tools (internal operations)
- [x] Implement environment-based URL resolution (localhost for dev, relative paths for prod)
- [x] Enable adding new tools by only editing config.yaml
- [x] All 37 BDD scenarios passing (6 new tools scenarios)
- [x] Frontend builds with no linter errors
- [x] Performance improved with local favicon serving (no external requests)

### Frontend Dynamic Filter Rendering (Oct 17, 2025)
- [x] Create ProcessorCapabilitiesService to fetch filter definitions from GetProcessorStatus
- [x] Refactor Filter interface to support dynamic FilterParameter arrays
- [x] Add createFilterFromDefinition() helper for proto-to-UI conversion
- [x] Update FilterPanel component to render parameters dynamically based on type
- [x] Replace hardcoded GRAYSCALE_ALGORITHMS with dynamic parameter options
- [x] Remove disabled filter logic (all filters from library are supported)
- [x] Initialize processor capabilities service in main.ts on app startup
- [x] Frontend adapts automatically when new filters are added to C++ library
- [x] All 31 BDD scenarios passing including new capabilities tests
- [x] Frontend builds with no linter errors

### API 2.0.0 & Dynamic Filter Metadata (Oct 17, 2025)
- [x] Upgrade processor API to version 2.0.0 (breaking change)
- [x] Add FilterDefinition proto with filter metadata (id, name, parameters)
- [x] Add FilterParameter proto for dynamic parameter discovery (type, options, default)
- [x] Refactor LibraryCapabilities to include full filter definitions
- [x] Rename ACCELERATOR_TYPE_GPU to ACCELERATOR_TYPE_CUDA for clarity
- [x] Add ACCELERATOR_TYPE_OPENCL for future OpenCL support
- [x] Implement processor_capabilities.feature with 4 BDD scenarios
- [x] Update BDD tests to validate filter definitions and parameters
- [x] Update all backend code to use ACCELERATOR_TYPE_CUDA
- [x] Fix processor_api_version() to use PROCESSOR_API_VERNUM define
- [x] Update config.yaml default_library from 1.0.0 to 2.0.0
- [x] Update start-dev.sh to generate API 2.0.0 metadata

### Plugin Architecture & Observability Stack (Oct 17, 2025)
- [x] Migrate from CGO direct binding to dlopen-based plugin system
- [x] Implement dynamic library loader with version compatibility checks
- [x] Add processor plugin API with versioned C ABI (processor_api.h)
- [x] Introduce processor registry for runtime plugin management
- [x] Add InitRequest/InitResponse with capability discovery
- [x] Implement mock_processor.cpp for GPU-less testing
- [x] Split monolithic proto into common, config_service, image_processor_service
- [x] Add library capabilities proto with filter/accelerator enumeration
- [x] Integrate Grafana with Loki and Promtail for log aggregation
- [x] Configure trace-to-logs correlation in Grafana dashboards
- [x] Add structured logging with Promtail multi-line parsing
- [x] Migrate Go webserver from Bazel to Makefile
- [x] Add VERSION file (1.0.0) for C++ accelerator API tracking
- [x] Add .prompts/ directory for AI-assisted testing workflows
- [x] Create comprehensive manual testing suite for multi-source grid
- [x] Simplify Docker documentation (remove redundant files)

### Multi-Source Video Processing (Oct 16, 2025)
- [x] Implement ListInputs endpoint with BDD test coverage
- [x] Create dynamic video grid component supporting up to 9 sources
- [x] Add drawer UI for source selection with real-time updates
- [x] Support per-source filter and resolution configuration
- [x] Implement frontend image scaling (original/half/quarter)
- [x] Modularize protobuf into separate service definitions (common, config_service, image_processor_service)
- [x] Add FAB controls for source management
- [x] Enhance UI with source indicators

### Integration Testing & BDD (Oct 15, 2025)
- [x] Implement godog/gherkin BDD framework with 31 passing scenarios
- [x] Add image_processing.feature with 14 scenarios (all filter/accelerator/grayscale combinations)
- [x] Add websocket_processing.feature with 4 scenarios
- [x] Add streaming_service.feature with 1 scenario
- [x] Add feature_flags.feature with 5 scenarios (GetStreamConfig, Sync, Health)
- [x] Add input_sources.feature with 5 scenarios
- [x] Add checksum validation using SHA-256 for image comparison
- [x] Dockerize integration tests with proper user permissions
- [x] Add cucumber HTML report visualization at port 5050
- [x] Migrate internal/ to pkg/ structure for better package visibility
- [x] Add /health endpoint with JSON response
- [x] Create automated test runner script (run-docker-tests.sh)

### Bug Fixes & Optimizations (Oct 15, 2025)
- [x] Fix missing CUDA error checks in grayscale_processor.cu
- [x] Fix dangling pointers from protobuf in cgo_api.cpp
- [x] Fix image decode error (PNG vs RAW pixels mismatch)
- [x] Add proper memory cleanup on CUDA errors
- [x] Remove duplicate buffer copies (-1MB per request, ~30% faster)
- [x] Remove unused code (intermediate_buffer, filter chaining)
- [x] Add strict compilation flags (-Werror=unused-*)

### Observability & Feature Flags (Oct 13, 2025)
- [x] Integrate Jaeger all-in-one for distributed tracing
- [x] Add OpenTelemetry collector for trace aggregation
- [x] Integrate Flipt for feature flag management
- [x] Implement automatic flag synchronization from YAML to Flipt on boot
- [x] Add service navigation buttons (Jaeger, Flipt, Sync Flags) in navbar
- [x] Create sync-flags-button Lit component with OpenTelemetry tracing
- [x] Add /api/flipt/sync REST endpoint for manual flag synchronization
- [x] Configure unified port mapping for Flipt services (8081 REST, 9000 gRPC)
- [x] Update start-dev.sh to verify service availability before startup

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
- **Backend**: Go + dlopen plugin system for C++/CUDA
- **Build**: Makefile (Go) + Bazel (C++/CUDA with C++23)
- **Frontend**: Lit Web Components + TypeScript + Vite
- **Testing**: GoogleTest/GMock (C++) + Godog BDD (Go)
- **Containerization**: Docker with NVIDIA Container Toolkit
- **Reverse Proxy**: Traefik with TLS termination
- **Observability**: Jaeger + OpenTelemetry + Grafana + Loki + Promtail
- **Feature Flags**: Flipt with automatic YAML synchronization

## Next Milestones

See individual backlog files for upcoming features:
- [kernels.md](./docs/backlog/kernels.md) - More image processing filters
- [video-streaming.md](./docs/backlog/video-streaming.md) - Optimized video transport
- [neural-networks.md](./docs/backlog/neural-networks.md) - CUDA neural networks
- [infrastructure.md](./docs/backlog/infrastructure.md) - Microservices and observability

