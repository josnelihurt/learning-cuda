# Infrastructure & DevOps

Microservices, observability, testing, and cloud deployment.

## gRPC Microservices

### Services to Build
- [x] Image Processing Service (ProcessImage endpoint implemented)
- [ ] StreamProcessVideo implementation (defined but returns Unimplemented)
- [ ] Video Management Service (ListVideos, StreamVideo, Upload)
- [ ] Model Inference Service (Predict, StreamInference)

### Connect-RPC (Instead of gRPC-Gateway)
- [x] Implemented Connect-RPC server
- [x] HTTP annotations in proto
- [x] Native HTTP/JSON support without gateway
- [ ] Implement bidirectional streaming for video
- [ ] Add Connect-Web for browser clients

### Infrastructure
- [x] Dynamic library loading with dlopen (plugin architecture)
- [x] Processor versioning and API compatibility checks
- [x] Processor registry with capability discovery
- [x] API 2.0.0 with FilterDefinition and FilterParameter metadata
- [x] Dynamic filter parameter discovery (type, options, defaults)
- [x] Split Go build system (Makefile) from C++ (Bazel)
- [ ] Service discovery (Consul optional)
- [ ] Load balancing (round-robin, retry, circuit breaker)
- [x] Clean arch: domain → application → infrastructure → interfaces
- [ ] Circuit breaker for Flipt calls with fallback
- [ ] Complete DI container (move processor loader from main.go)
- [ ] Connection pooling monitoring and metrics
- [ ] Plugin hot-reloading without server restart
- [ ] Multiple processor plugins loaded simultaneously

## Observability

### Jaeger (Distributed Tracing)
- [x] Add Jaeger to docker-compose
- [x] OpenTelemetry SDK for Go
- [x] OpenTelemetry SDK for frontend (browser)
- [x] Distributed tracing from browser to backend
- [x] Trace context propagation with W3C headers
- [ ] Trace WebSocket, CGO calls, CUDA kernels
- [ ] Span attributes: width, height, filter type
- [ ] Enhanced trace attributes (user-agent, IP, request size)
- [ ] Baggage for cross-service context
- [ ] Intelligent trace sampling based on load

### Prometheus (Metrics)
- [ ] Add Prometheus to docker-compose
- [ ] Expose /metrics endpoint
- [ ] Custom metrics: frames_processed, duration, connections
- [ ] GPU metrics exporter
- [ ] Processing latency histogram by accelerator and filter
- [ ] Error counters by type and handler
- [ ] Active WebSocket connections gauge

### Grafana (Dashboards)
- [ ] Connect Prometheus + Jaeger
- [ ] Dashboard 1: FPS, latency, GPU vs CPU
- [ ] Dashboard 2: GPU utilization, memory, temp
- [ ] Dashboard 3: Request rate, errors, latency
- [ ] Alerts: high errors, latency, GPU temp

### Feature Flags
- [x] Integrate Flipt for feature flag management
- [x] Automatic flag synchronization from YAML to Flipt
- [x] REST API endpoint for manual flag sync
- [x] Web UI integration with sync button
- [x] Fallback to YAML when Flipt unavailable
- [ ] Advanced flag rules and targeting

### Logging
- [x] spdlog (C++)
- [x] Health endpoint at /health with JSON response
- [x] Add trace_id to logs
- [x] Replace Go log package with structured logging (zerolog)
- [x] Loki stack deployed (Loki + Promtail + Grafana)
- [x] Trace-to-Logs correlation in Grafana
- [x] Consistent log levels (debug/info/warn/error)
- [x] Multi-line log parsing in Promtail
- [x] Grafana dashboard and datasource provisioning
- [ ] Add request correlation IDs across services
- [ ] Detailed health checks with dependencies (readiness vs liveness)

## Load Testing & BDD

### Unit Tests
- [x] Setup test coverage reporting infrastructure
- [x] Add unit tests for ProcessImageUseCase with mocks (12 test cases, 211 lines)
- [x] Add unit tests for EvaluateFeatureFlagUseCase (7 test cases, 240 lines)
- [x] Add unit tests for GetStreamConfigUseCase (3 test cases, 113 lines)
- [x] Add unit tests for ListInputsUseCase (2 test cases, 87 lines)
- [x] Add unit tests for SyncFeatureFlagsUseCase (3 test cases, 110 lines)
- [x] Add unit tests for ListAvailableImagesUseCase (2 test cases, 96 lines)
- [x] Add unit tests for StaticImageRepository infrastructure (2 test cases, 99 lines)
- [x] Add unit tests for ToastContainer frontend component (8 test suites, 199 lines)
- [x] Add unit tests for ImageSelectorModal frontend component (8 test suites, 128 lines)
- [x] Create mocks for FeatureFlagRepository interface
- [x] Create mocks for ImageProcessor interface
- [x] Create mocks for StaticImageRepository interface
- [x] Total application layer coverage: 6 use cases tested (29+ test cases, 857 lines)
- [x] Total infrastructure layer coverage: 1 repository tested (99 lines)
- [ ] Add unit tests for handlers (ConnectRPC, WebSocket, HTTP)
- [ ] Add unit tests for repositories (mock Flipt client)
- [ ] Add unit tests for remaining infrastructure services
- [ ] Mock all remaining domain interfaces

### K6 Load Tests
- [ ] Single user baseline (30 FPS for 30s)
- [ ] Concurrent users (ramp to 100, sustain, ramp down)
- [ ] Stress test (find breaking point)
- [ ] gRPC tests with `ghz`

### Godog BDD
- [x] Setup Godog + feature files
- [x] Feature: Image processing (14 scenarios)
- [x] Feature: WebSocket processing (4 scenarios)
- [x] Feature: Streaming service (1 scenario)
- [x] Feature: Feature flags (5 scenarios)
- [x] Feature: Input sources (3 scenarios)
- [x] Feature: Processor capabilities (4 scenarios)
- [x] Feature: Tools configuration (6 scenarios)
- [x] Feature: Available images (3 scenarios)
- [x] Step definitions in Go
- [x] CI integration with dockerized tests
- [x] Cucumber HTML reports at port 5050
- [x] 40 scenarios passing with dynamic filter, tools, and image selection validation

### Playwright E2E Tests (Frontend)
- [x] Setup Playwright with TypeScript
- [x] Configure multi-browser testing (Chrome, Firefox, WebKit)
- [x] Test: Drawer functionality (4 tests)
- [x] Test: Filter configuration (5 tests)
- [x] Test: Filter toggle behavior (4 tests)
- [x] Test: Multi-source management (4 tests)
- [x] Test: Panel synchronization (5 tests)
- [x] Test: Resolution control (3 tests)
- [x] Test: Source removal (2 tests)
- [x] Test: UI validation (8 tests)
- [x] Test: WebSocket management (2 tests)
- [x] Test: Image selection (8 tests)
- [x] Create reusable test helpers
- [x] Add data-testid attributes to components
- [x] Docker integration for CI/CD
- [x] Test artifacts and HTML reports
- [x] Total: 10 test suites with 45 E2E tests
- [ ] Visual regression testing with screenshots
- [ ] Performance testing with Lighthouse
- [ ] Accessibility testing with axe-core
- [ ] Mobile viewport testing

## Cloud Deployment

### Provider Research
- [ ] Vultr (A40, A100 pricing)
- [ ] Lambda Labs, Paperspace, AWS P3/G4, GCP
- [ ] Create comparison doc: cost, availability, latency
- [ ] Document choice with reasoning

### Terraform
- [ ] Setup IaC for cloud resources
- [ ] Cloud-init: Docker, NVIDIA toolkit, SSL
- [ ] State management (Terraform Cloud or S3)

### CI/CD
- [ ] GitHub Actions for build/test
- [ ] Auto deploy on merge to main
- [ ] Deployment strategy: blue-green, rolling, or canary

### Production Monitoring
- [ ] Cloud monitoring (CloudWatch/Stackdriver)
- [ ] Alerts (PagerDuty, Slack)
- [ ] Log aggregation

### Security
- [ ] Secrets management (Vault, AWS Secrets Manager)
- [x] TLS for HTTP server (native Go implementation)
- [ ] TLS for gRPC
- [ ] Auth (JWT, OAuth)
- [ ] Rate limiting
- [ ] DDoS protection (Cloudflare)
- [ ] Fix WebSocket CORS validation (currently accepts all origins)
- [ ] Input validation layer (image size limits, request validation)
- [ ] Custom domain errors with proper error codes

## Code Quality

### Linters & Formatters
- [x] Configure clang-tidy for C++ with 50+ code quality checks
- [x] Configure golangci-lint with 20+ linters for Go
- [x] Configure ESLint + Prettier for TypeScript/Lit frontend
- [x] Create unified run-linters.sh script (supports --fix mode)
- [x] Add Docker services for all linters (lint-frontend, lint-golang, lint-cpp)
- [x] Enhanced .clang-format with comprehensive formatting rules
- [x] Configure linter exclusions for tests, proto, and generated code
- [ ] Pre-commit hooks for automatic linting
- [ ] CI/CD integration for linting on pull requests

### Testing & Coverage
- [x] Setup Vitest for frontend unit tests with coverage
- [x] Create run-coverage.sh script for all test coverage
- [x] Add coverage-report-viewer Docker service (port 5052)
- [x] Configure coverage thresholds (80% for all metrics)
- [x] Implement Go unit tests for ProcessImageUseCase (12 test cases)
- [x] Implement frontend unit tests for ToastContainer (8 test suites)
- [x] Coverage reports: Frontend (HTML/JSON/LCOV), Go (HTML), C++ (LCOV)
- [ ] Increase Go test coverage to 80%
- [ ] Add unit tests for all use cases
- [ ] Add unit tests for all handlers
- [ ] Add unit tests for repositories
- [ ] Integration tests for processor plugins

### Go Backend
- [x] Extract processor loading logic into separate loader package
- [x] Implement version compatibility validation
- [x] Add proper error handling for plugin lifecycle
- [x] Refactor Viper configuration to use mapstructure tags
- [x] Add environment variable for mock mode in testing
- [x] Simplify config initialization with automatic unmarshaling
- [ ] Add godoc documentation for all exported types/functions
- [ ] Extract magic numbers to constants (timeouts, intervals, buffer sizes)
- [ ] Refactor long functions (websocket.processFrame)
- [ ] Add validation DTOs (separate from domain models)
- [ ] Error handling middleware for HTTP/gRPC
- [ ] Add tools.go for build dependencies versioning
- [ ] Configuration validation at startup (fail-fast)
- [ ] Organize config files by environment (dev, prod, test)

### Input Sources
- [x] Load static images dynamically from /data directory
- [x] Implement StaticImageRepository with filesystem scanning
- [x] Support .png, .jpg, .jpeg image formats
- [x] Add UI for selecting and changing static images
- [x] ListAvailableImages endpoint with image metadata
- [ ] Support multiple server-side cameras
- [ ] Support video files as input sources
- [ ] Support remote stream URLs (RTSP, HLS)
- [ ] Implement GetFlag in FliptRepository

## Notes

Start with observability first. Keep it simple initially. GPU instances are expensive, monitor costs.

