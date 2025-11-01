# Changelog

Completed features extracted from git commit history, organized by date.

## November 2025

### Production Deployment & Infrastructure (Nov 1, 2025)
- [x] Implemented production Docker deployment with CUDA separation
- [x] Added Cloudflare tunnel integration for production environment on Jetson Nano
- [x] Created modular Jetson Nano deployment system with Ansible automation
- [x] Unified logging configuration for development and production environments
- [x] Added production deployment warning banner with environment detection
- [x] Fixed Flipt repository sync issues in production environment
- [x] Enhanced camera preview functionality with video grid filter improvements
- [x] Fixed E2E tests for production compatibility

**Production URLs:**
- Main Application: https://app-cuda-demo.josnelihurt.me
- Grafana Monitoring: https://grafana-cuda-demo.josnelihurt.me
- Feature Flags: https://flipt-cuda-demo.josnelihurt.me
- Distributed Tracing: https://jaeger-cuda-demo.josnelihurt.me
- Test Reports: https://reports-cuda-demo.josnelihurt.me

**Deployment Features:**
- Cloudflare tunnel for secure external access
- Ansible playbooks for automated deployment and updates
- Production-optimized Docker configuration with CUDA separation
- Environment-specific configuration management
- Automated code synchronization from development to production

### Connect-RPC GET Request Support & Routing Fix (Nov 1, 2025)

**Issue Summary:**
- Connect-RPC methods with `idempotency_level = NO_SIDE_EFFECTS` were returning `405 Method Not Allowed` for GET requests
- Despite correct proto configuration, generated handlers, and client-side `useHttpGet: true`, direct Connect-RPC GET calls failed
- Frontend configured with `useHttpGet: true` in Connect-RPC transport but GET requests were rejected by server

**Investigation Process:**
- [x] Verified `idempotency_level = NO_SIDE_EFFECTS` present in all GET methods in proto files
- [x] Confirmed generated handlers include `connect.WithIdempotency(connect.IdempotencyNoSideEffects)`
- [x] Upgraded Connect-RPC from v1.17.0 to v1.19.1 to ensure GET support
- [x] Updated Go version in bufgen.dockerfile from 1.23 to 1.24 for compatibility
- [x] Regenerated proto code with updated protoc-gen-connect-go
- [x] Reviewed Connect-RPC source code to verify GET support logic in `protocol_connect.go`
- [x] Created minimal test case confirming handlers support GET when created directly
- [x] Traced routing flow to identify catch-all handler interception

**Root Cause:**
The catch-all handler in `webserver/pkg/app/app.go` was intercepting Connect-RPC routes (`/cuda_learning.*`, `/com.jrb.*`) and forwarding them to Vanguard transcoder before they reached the registered Connect-RPC handlers. Vanguard transcoder is designed for REST-to-Connect-RPC transcoding, not for handling direct Connect-RPC protocol requests. As a result, GET requests for Connect-RPC endpoints were routed incorrectly and rejected with 405.

**Solution:**
- [x] Removed fallback to Vanguard transcoder for Connect-RPC routes in catch-all handler
- [x] Ensured Connect-RPC handlers are registered before catch-all to handle both GET and POST
- [x] Catch-all now only handles REST API routes (`/api/*`) and SPA index
- [x] Connect-RPC routes return 404 if no handler matches (instead of falling back to transcoder)

**Technical Details:**
- Connect-RPC handlers support GET automatically when `IdempotencyLevel == IdempotencyNoSideEffects` and `StreamType == StreamTypeUnary`
- The generated handler wrapper correctly applies idempotency options
- HTTP method routing in Connect-RPC is handled by protocol handlers, not HTTP mux routing
- Vanguard transcoder should only handle REST endpoints, not direct Connect-RPC protocol calls

**Lessons Learnt:**
1. **Routing Order Matters**: When using multiple handlers (Connect-RPC, Vanguard, SPA), ensure specific routes are registered before catch-all handlers
2. **Transcoder Scope**: Vanguard transcoder is for REST-to-RPC transcoding, not for direct Connect-RPC protocol handling
3. **Minimal Tests Help**: Creating isolated test cases helped identify that the issue was routing, not handler configuration
4. **Source Code Review**: When documentation is unclear, reviewing library source code provides definitive answers about expected behavior
5. **Protocol vs Transport**: Understanding the distinction between Connect-RPC protocol (handled by registered handlers) and REST transcoding (handled by Vanguard) is critical for proper routing

**Files Changed:**
- `webserver/pkg/app/app.go`: Fixed catch-all handler routing logic
- `bufgen.dockerfile`: Updated Go 1.24 and Connect-RPC v1.19.1
- `proto/config_service.proto`: Added `idempotency_level = NO_SIDE_EFFECTS` to GET methods
- `proto/file_service.proto`: Added `idempotency_level = NO_SIDE_EFFECTS` to GET methods
- Frontend services: Added `useHttpGet: true` to Connect-RPC transport configuration

### Staging Environment (Oct 31, 2025)
- [x] Added localhost staging environment with docker-compose.staging.yml
- [x] Configured Traefik reverse proxy for staging with HTTPS auto-redirect
- [x] Created staging-local deployment scripts (start.sh, stop.sh, clean.sh)
- [x] Services accessible via .localhost domains (app.localhost, grafana.localhost, etc.)
- [x] Production-like Docker deployment for local testing
- [x] Staging configuration file (config.staging.yaml) with appropriate settings

**Staging URLs:**
- Main Application: https://app.localhost
- Grafana Monitoring: https://grafana.localhost
- Feature Flags (Flipt): https://flipt.localhost
- Distributed Tracing (Jaeger): https://jaeger.localhost
- Test Reports: https://reports.localhost

**Usage:**
```bash
./scripts/deployment/staging_local/start.sh        # Start staging environment
./scripts/deployment/staging_local/stop.sh         # Stop services
./scripts/deployment/staging_local/clean.sh        # Clean volumes and images
```

### Git Hooks & Testing (Oct 31, 2025)
- [x] Fixed pre-push hook to use correct server flags
- [x] Improved hook execution for test validation

### Deployment Fixes (Oct 30, 2025)
- [x] Various deployment configuration fixes
- [x] Improved deployment reliability

### Documentation (Oct 27, 2025)
- [x] Updated project documentation
- [x] Improved documentation structure and clarity

### UI Improvements (Oct 26, 2025)
- [x] Refactored production banner to information banner with simplified logic
- [x] Fixed camera preview with video grid filter application
- [x] Improved video processing integration
- [x] Fixed E2E tests for camera functionality

### System Information & Feature Flags (Oct 26, 2025)
- [x] Added GetSystemInfo endpoint with comprehensive version information
- [x] Created version tooltip component with build integration
- [x] Implemented feature flags modal with Flipt iframe integration
- [x] Enhanced system information display with backend data integration
- [x] Added telemetry integration for system info operations
- [x] Created comprehensive BDD scenarios for system info validation

**Benefits:**
- Centralized system information management
- Real-time version tracking across all components
- Enhanced observability with telemetry integration
- Improved user experience with interactive tooltips
- Production-ready feature flag management interface

## October 2025

### GetSystemInfo RPC Endpoint (Oct 25, 2025) - 5 BDD scenarios
- [x] Added GetSystemInfo RPC endpoint in ConfigService
- [x] Created SystemVersion proto message with 6 version fields (C++, Go, JS, Branch, Build, Commit)
- [x] Implemented domain models for SystemInfo and SystemVersion
- [x] Created build info infrastructure layer with environment variable support
- [x] Implemented GetSystemInfoUseCase with observability and telemetry
- [x] Added GetSystemInfo handler with span attributes and error recording
- [x] Wired GetSystemInfoUseCase through container, app, and server
- [x] Created system-info-service.ts with telemetry integration
- [x] Updated version-tooltip-lit component to use backend endpoint
- [x] Added 5 BDD scenarios for system info validation
- [x] Created E2E tests for version tooltip functionality

**Benefits:**
- Consolidated system information in single endpoint
- Replaced hardcoded version values with backend data
- Full observability with telemetry spans and attributes
- Clean Architecture with proper separation of concerns
- Comprehensive test coverage across all layers

**Test Coverage:**
- Backend: Unit tests for use case, handler, and build info
- Frontend: E2E tests for version tooltip component
- BDD: 5 scenarios covering all version fields and processor status

### Video Frame ID and Test Infrastructure (Oct 20, 2025) - 6 BDD scenarios (pending implementation)
- [x] Generated optimized E2E test video (480x360, 10fps, 20s, 200 frames, 464KB)
- [x] Extracted 200 frames as PNG with SHA256 hash metadata
- [x] Created embedded Go metadata (webserver/pkg/infrastructure/video/test_video_metadata.go)
- [x] Extended VideoFrameUpdate proto with frame_id field
- [x] Updated video streaming to include sequential frame_id (0-199)
- [x] Created scripts for video generation and frame extraction
- [x] Added helper functions: GetFrameMetadata(), ValidateFrameHash()
- [x] Created BDD feature file with 6 scenarios for frame_id validation
- [x] Implemented stub BDD step definitions (awaiting WebSocket frame collection)
- [x] Added E2E Playwright test for frame_id sequential validation
- [x] Fixed frame_id capture in E2E tests using console.log parsing
- [x] Updated all video E2E tests to use e2e-test.mp4 instead of sample.mp4
- [x] Adjusted pixel difference threshold from 50% to 10% for smaller video
- [x] Fixed Firefox preview loading with proper wait states
- [x] Generated preview image for e2e-test.mp4
- [x] Excluded PNG frames from repository (62MB, generated on-demand)
- [x] Updated scripts to verify existence before generating frames
- [x] Added automatic frame extraction in metadata generation tool
- [x] Updated .gitignore to include test video but exclude PNG frames
- [x] Documented on-demand frame generation process in README

**Benefits:**
- Small test video suitable for repository (<500KB vs 158MB)
- Deterministic frame validation with pre-calculated hashes
- Better E2E test reliability with frame_id tracking
- Metadata versioned as code for easy access

**Test Infrastructure:**
- Video: data/test-data/videos/e2e-test.mp4
- Frames: data/test-data/video-frames/e2e-test/*.png (200 frames)
- Metadata: Embedded in webserver/pkg/infrastructure/video/test_video_metadata.go
- Scripts: generate-test-video.sh, extract-video-frames.sh, generate-video-metadata

### WebSocket Handler Refactoring (Oct 20, 2025)
- [x] Commented non-functional video player code pending FFmpeg integration
- [x] Added TODO comments documenting planned migration from WebSocket to gRPC streaming
- [x] Updated trace proxy handler to return JSON success when tracing disabled
- [x] Fixed linter issues in websocket and http handlers
- [x] Prepared video session management infrastructure for future implementation

**Migration Path**: WebSocket → Connect-RPC bidirectional streaming
- Documented in: `webserver/pkg/interfaces/websocket/handler.go`
- Target: `webserver/pkg/interfaces/connectrpc/handler.go StreamProcessVideo`
- Benefits: Type-safe protocol, unified API, better error handling

### Video Playback Infrastructure (Oct 19, 2025) - 9 BDD scenarios
- [x] Extended FileService proto with video upload/list RPCs (ListAvailableVideos, UploadVideo)
- [x] Created StaticVideo message in common.proto with preview image support
- [x] Extended InputSource with video_path and preview_image_path fields
- [x] Implemented VideoRepository interface and FileVideoRepository
- [x] Created VideoPlayer interface and GoVideoPlayer
- [x] Implemented ListVideosUseCase with telemetry
- [x] Implemented UploadVideoUseCase with validation
- [x] Created VideoPlaybackUseCase for frame processing coordination
- [x] Extended ListInputsUseCase to include video sources
- [x] Extended FileHandler with ListAvailableVideos and UploadVideo RPCs
- [x] Updated ConfigHandler to support video fields in InputSource
- [x] Integrated video components in DI container, app, and main
- [x] Created VideoService frontend with upload/list methods
- [x] Implemented VideoSelector component with preview grid
- [x] Implemented VideoUpload component with drag-drop
- [x] Extended SourceDrawer with Images/Videos tabs
- [x] Added WebSocket VideoSessionManager for frame streaming
- [x] Created BDD feature file with 9 scenarios for video functionality
- [x] Implemented BDD step definitions for video upload/list/validation
- [x] Created E2E tests for video UI components (6 test cases)
- [x] MP4 format support with size validation (100MB limit)
- [x] Telemetry integration for video operations

### Image Upload with FileService (Oct 18, 2025)
- [x] Create FileService proto with ListAvailableImages and UploadImage RPCs
- [x] Refactor ListAvailableImages from ConfigService to FileService
- [x] Implement FileHandler with upload and list capabilities
- [x] Create UploadImageUseCase with file validation
- [x] Add PNG format validation (magic number check)
- [x] Enforce 10MB file size limit with validation
- [x] Extend StaticImageRepository with Save method
- [x] Add repository unit tests for Save method (2 test cases)
- [x] Create ImageUpload web component with drag-and-drop
- [x] Implement upload progress indicator with visual feedback
- [x] Add file validation in frontend (format and size)
- [x] Display user-friendly error messages for validation failures
- [x] Integrate upload component in SourceDrawer
- [x] Update InputSourceService to use FileService client
- [x] Create FileService TypeScript client with tracing
- [x] Add 4 BDD scenarios for upload validation in upload_images.feature
- [x] Implement comprehensive BDD step definitions
- [x] Create UploadImageUseCase unit tests with 6 test cases
- [x] Add frontend unit tests for ImageUpload component
- [x] Create E2E tests for upload workflow
- [x] Wire FileService in server registration and container
- [x] Update main.go with UploadImageUseCase dependency
- [x] Add OpenTelemetry tracing to all upload operations
- [x] Uploaded images saved to /data/static_images directory

### Code Quality & Modular Git Hooks (Oct 18, 2025)
- [x] Modularize git hooks into separate scripts per layer
- [x] Create pre-commit-cpp.sh for C++ tests
- [x] Create pre-commit-go.sh for Go tests
- [x] Create pre-commit-frontend.sh for frontend tests
- [x] Create pre-commit-lint-cpp.sh for C++ linting
- [x] Create pre-commit-lint-go.sh for golangci-lint
- [x] Create pre-commit-lint-frontend.sh for TypeScript linting
- [x] Refactor main pre-commit hook to orchestrate modular scripts
- [x] Update .golangci.yml to disable deprecated exportloopref linter
- [x] Configure govet to disable fieldalignment, enable shadow checks
- [x] Add gosec exceptions for G114 (HTTP timeouts) and G115 (integer conversions)
- [x] Add path-based linter exclusions for development_handler.go
- [x] Rename static_http package to statichttp (Go naming convention)
- [x] Remove unused interfaces.go file
- [x] Clean up unused imports across 20+ Go files
- [x] Fix code formatting and linter violations
- [x] Update BDD test context and assertions
- [x] Update docker-compose.dev.yml configuration

### Git Hooks & C++ Test Quality (Oct 18, 2025)
- [x] Add pre-commit git hook with C++ tests, Go tests, linters, and frontend build
- [x] Add pre-push git hook with full validation across all browsers
- [x] Create setup-hooks.sh script for easy installation
- [x] Refactor C++ tests to follow AAA pattern (Arrange/Act/Assert)
- [x] Improve test readability with consistent indentation
- [x] Update image paths in tests to use static_images directory
- [x] Fix command_factory_test.cpp formatting
- [x] Fix config_manager_test.cpp formatting and paths
- [x] Fix image_loader_test.cpp formatting and expectations
- [x] Update BUILD files with proper indentation
- [x] Update README with git hooks documentation and usage
- [x] Support --no-verify flag to skip hooks when needed

### Dynamic Static Image Selection (Oct 18, 2025)
- [x] Implement StaticImage domain model with ID, DisplayName, Path, IsDefault fields
- [x] Create StaticImageRepository interface in domain layer
- [x] Implement filesystem-based StaticImageRepository with directory scanning
- [x] Support .png, .jpg, and .jpeg image formats
- [x] Add ListAvailableImagesUseCase with OpenTelemetry tracing
- [x] Add ListAvailableImages RPC endpoint to ConfigService proto
- [x] Create ImageSelectorModal web component with animated UI
- [x] Add "Change Image" button to static video source cards
- [x] Implement modal with image grid, preview thumbnails, and selection
- [x] Add BDD feature: available_images with 3 scenarios
- [x] Implement 8 Playwright E2E tests for image selection flow
- [x] Implement 8 Vitest unit tests for ImageSelectorModal
- [x] Add infrastructure tests for StaticImageRepository
- [x] Add application layer tests for ListAvailableImagesUseCase
- [x] Configure static_images directory in config.yaml
- [x] Add images: lena, mandrill, peppers, barbara to data directory
- [x] Update Docker configuration to mount static images
- [x] Total: 19 tests (3 BDD + 8 E2E + 8 unit) across 4 test files

### Go Application Layer Unit Tests (Oct 18, 2025)
- [x] Implement EvaluateFeatureFlagUseCase tests with 7 comprehensive test cases
- [x] Test boolean flag evaluation with success and error scenarios
- [x] Test variant flag evaluation with multiple variant types
- [x] Implement GetStreamConfigUseCase tests with 3 test cases
- [x] Test stream configuration retrieval and error handling
- [x] Implement ListInputsUseCase tests with 2 test cases
- [x] Test input source listing with service integration
- [x] Implement SyncFeatureFlagsUseCase tests with 3 test cases
- [x] Test feature flag synchronization with Flipt repository
- [x] Create comprehensive mocks for FeatureFlagRepository interface
- [x] Add table-driven test patterns for maintainability
- [x] Total: 9 test functions with 15+ test cases across 4 use cases

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
- [x] Implement Go unit tests for ProcessImageUseCase with 12 test cases
- [x] Implement frontend unit tests for ToastContainer with 8 test suites
- [x] Configure coverage thresholds (80% for lines, functions, branches, statements)
- [x] Add coverage reports for frontend (HTML, JSON, LCOV), Golang (HTML), and C++ (LCOV)
- [x] Update README with Testing & Code Quality section
- [x] Enhanced .clang-format configuration with 68 formatting rules
- [x] Total test coverage infrastructure across 3 languages

### End-to-End Testing with Playwright (Oct 17, 2025)
- [x] Integrate Playwright testing framework with TypeScript
- [x] Add 9 comprehensive E2E test suites
- [x] Implement drawer-functionality tests (4 tests)
- [x] Implement filter-configuration tests (5 tests)
- [x] Implement filter-toggle tests (4 tests)
- [x] Implement multi-source-management tests (4 tests)
- [x] Implement panel-synchronization tests (5 tests)
- [x] Implement resolution-control tests (3 tests)
- [x] Implement source-removal tests (2 tests)
- [x] Implement ui-validation tests (8 tests)
- [x] Implement websocket-management tests (2 tests)
- [x] Create test-helpers utility with reusable functions
- [x] Add data-testid attributes to all interactive frontend components
- [x] Configure Playwright for Chrome, Firefox, and WebKit testing
- [x] Create run-e2e-tests.sh script for automated test execution
- [x] Integrate E2E tests with Docker Compose for CI/CD
- [x] Add Playwright service to docker-compose.dev.yml
- [x] Configure test artifacts and reports output
- [x] Add npm scripts for running tests (headed/headless/debug/UI modes)
- [x] Update .gitignore for Playwright artifacts

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

### Frontend Development (Oct 10-11, 2025)
- [x] Enable dev mode with hot reload in start-dev.sh script
- [x] Redesign UI to dashboard layout with fixed sidebar and stats footer
- [x] Adjust dashboard UI spacing and filter auto-expand behavior
- [x] Modularize frontend architecture with toast notifications and auto-versioning
- [x] Simplify shell scripts to minimal professional style
- [x] Remove verbose comments and emojis from codebase

### Image Processing (Oct 10, 2025)
- [x] Add multi-filter support with GPU/CPU selection
- [x] Implement multiple grayscale algorithms (BT.601, BT.709, Average, Lightness, Luminosity)

### HTTPS & Security (Oct 10, 2025)
- [x] Add HTTPS support with Caddy reverse proxy for local development
- [x] Configure Caddy reverse proxy with port 8443 for local HTTPS

### Documentation (Oct 10, 2025)
- [x] Add project README with setup and usage instructions
- [x] Add application screenshot

### Backend & Integration (Oct 9, 2025)
- [x] Implement web server with image filter UI and WebSocket support
- [x] Implement CGO integration with CUDA and real-time webcam streaming
- [x] Optimize video streaming performance and add resolution selector

### Build System (Oct 9, 2025)
- [x] Integrate Golang with Bazel build system
- [x] Add Docker support with Bazel OCI rules and simplify Go build files

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
