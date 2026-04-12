# External Integrations

**Analysis Date:** 2026-04-12

## APIs & External Services

**Observability & Monitoring:**
- **Jaeger Tracing** - Distributed tracing for Go, C++, and frontend
  - Endpoint: `https://jaeger-cuda-demo.josnelihurt.me`
  - SDK: OpenTelemetry (Go: v1.38.0, C++: v1.18.0, Frontend: v1.28.0)
  - Protocol: OTLP over gRPC (port 4317) and HTTP (port 4318)
  - Auth: None (internal network)

- **Grafana** - Monitoring dashboards and log visualization
  - URL: `https://grafana-cuda-demo.josnelihurt.me`
  - Dashboards: Multi-layer logs, system metrics
  - Data sources: Loki (logs), Jaeger (traces)
  - Auth: None configured (internal access)

- **Loki** - Log aggregation
  - Endpoint: `http://loki-cloud:3100/loki/api/v1/push` (staging/production)
  - Protocol: HTTP
  - Integration: OpenTelemetry Collector -> Loki
  - Auth: None (internal network)

**Feature Flags:**
- **Flipt** - Feature flag management
  - URL: `http://localhost:8081` (dev), `http://flipt:8080` (staging/production)
  - SDK: go.flipt.io/flipt-client v1.2.0
  - Storage: SQLite database (`flipt.db`)
  - Auth: None (internal service)
  - Managed via HTTP API and Go SDK
  - Flags: binary_protocol, grpc_streaming, observability features

**IoT Device Management:**
- **MQTT Broker** - Remote device power management
  - Broker: `vultur.josnelihurt.me` (dev), `mosquitto` (staging/production)
  - Port: 1883
  - SDK: github.com/eclipse/paho.mqtt.golang v1.5.1
  - Topics: `pow/S31JetsonNanoOrin` (power monitoring), `pow/+/SENSOR` (sensor data)
  - Auth: None configured
  - Purpose: Monitor and control Jetson Nano power consumption

**Tunneling & Reverse Proxy:**
- **Cloudflare Tunnel** - Secure external access
  - Tunnel ID: ef1d45eb-5e53-4a40-8bc3-e30c1e29e00d
  - Service: cloudflared (Docker container)
  - Domains:
    - app-cuda-demo.josnelihurt.me
    - grafana-cuda-demo.josnelihurt.me
    - flipt-cuda-demo.josnelihurt.me
    - jaeger-cuda-demo.josnelihurt.me
    - reports-cuda-demo.josnelihurt.me
    - grpc.app-cuda-demo.josnelihurt.me
  - Auth: Cloudflare account authentication
  - Origin: Traefik reverse proxy

## Data Storage

**Databases:**
- **SQLite** - Flipt feature flags storage
  - Connection: Local file system
  - Path: `.ignore/storage/flipt/flipt.db` (dev), `/var/opt/flipt/flipt.db` (staging/production)
  - Client: Flipt internal SQLite client
  - Managed by: Flipt service

**File Storage:**
- **Local filesystem** - Static assets and test data
  - Directory: `data/static_images/` for sample images
  - Directory: `data/videos/` for video files
  - Directory: `data/test-data/` for test fixtures
  - Served by: Go HTTP server
  - No external storage service used

**Caching:**
- None - No caching layer implemented

## Authentication & Identity

**Auth Provider:**
- **Custom TLS-based authentication** - No external auth provider
  - Implementation: Self-signed certificates for development
  - Production: TLS termination at Traefik
  - No OAuth/OIDC integration
  - No user authentication system (application is public)
  - Authorization: Not applicable (public application)

**TLS Certificates:**
- Development: Self-signed certificates in `.secrets/`
- Production: Certificates managed by Traefik
- Configuration: `server.tls` section in config files

## Monitoring & Observability

**Error Tracking:**
- **OpenTelemetry Collector** - Centralized telemetry collection
  - Config: `otel-collector-config.yaml`
  - Receivers: OTLP (gRPC + HTTP)
  - Processors: batch, memory_limiter, resource enrichment
  - Exporters: Jaeger (traces), Loki (logs), logging (debug)
  - Health check: `0.0.0.0:13133`

**Logs:**
- **Structured logging** with zerolog (Go), spdlog (C++)
- **Log levels:** debug, info, warn, error
- **Formats:** JSON (production), console (development)
- **Outputs:** stdout, file (/tmp/goserver.log in production), remote OTLP endpoint
- **Remote logging:** `https://otel-cuda-demo.josnelihurt.me/v1/logs`
- **Log aggregation:** Loki via OpenTelemetry Collector

**Metrics:**
- No dedicated metrics collection (e.g., Prometheus)
- Performance metrics collected via OpenTelemetry traces
- GPU/CPU performance comparison built into application

## CI/CD & Deployment

**Hosting:**
- **GitHub Container Registry (GHCR)** - Docker image registry
  - Registry: `ghcr.io/josnelihurt/learning-cuda`
  - Images: grpc-server, go-app, test-reports, intermediate build artifacts
  - Auth: GitHub token (GITHUB_TOKEN)

**CI Pipeline:**
- **GitHub Actions** - CI/CD automation
  - Workflows:
    - `.github/workflows/docker-monorepo-build-arm.yml` - ARM64 builds (Radxa)
    - `.github/workflows/docker-monorepo-build-x86.yml` - x86 builds
  - Platforms: ARM64 (Jetson Nano, Radxa), x86 (cloud VM)
  - Self-hosted runners: ARM64 (radxa, jetson-nano)
  - Automated builds on push to main
  - Automated deployment to Jetson Nano on main branch push

**Deployment:**
- **Docker Compose** - Multi-container orchestration
  - Files: `docker-compose.yml`, `docker-compose.staging.yml`, `docker-compose.dev.yml`
  - Services: app, grpc-server, traefik, flipt, otel-collector, cloudflared
  - Runtime: NVIDIA Container Toolkit for GPU access
  - Deployment targets: Local (dev), Staging (Docker), Production (Jetson Nano)

**Container Registry:**
- Intermediate images stored in GHCR for multi-stage builds
- Versioned images for reproducible deployments
- Architecture-specific tags (arm64, amd64)

## Environment Configuration

**Required env vars:**

**Go Server (from YAML config):**
- `environment` - Environment name (development/staging/production)
- `server.http_port` - HTTP port
- `server.https_port` - HTTPS port
- `server.hot_reload_enabled` - Enable Vite dev server
- `server.web_root_path` - Frontend static files path
- `processor.library_base_path` - C++ shared library path
- `processor.grpc_server_address` - gRPC server address
- `flipt.url` - Flipt service URL
- `flipt.namespace` - Flipt namespace
- `observability.otel_collector_grpc_endpoint` - OTLP gRPC endpoint
- `observability.otel_collector_http_endpoint` - OTLP HTTP endpoint
- `mqtt.broker` - MQTT broker address
- `mqtt.port` - MQTT port
- `mqtt.client_id` - MQTT client identifier
- `mqtt.topic` - MQTT topic for power monitoring

**Frontend (Vite):**
- `__APP_VERSION__` - Git commit hash (injected at build time)
- `__APP_BRANCH__` - Git branch name (injected at build time)
- `__BUILD_TIME__` - Build timestamp (injected at build time)
- `PLAYWRIGHT_BASE_URL` - Base URL for E2E tests
- `NODE_ENV` - Node environment (development/production)

**Secrets location:**
- Development: `.secrets/` directory (gitignored)
- Production: Environment variables in Docker Compose
- CI/CD: GitHub Secrets and Variables
- MQTT broker credentials: Environment variables
- Cloudflare tunnel token: `.secrets/production.env`
- TLS certificates: `.secrets/localhost+2.pem` (dev), Traefik-managed (production)

## Webhooks & Callbacks

**Incoming:**
- None - No webhook endpoints implemented

**Outgoing:**
- None - No outgoing webhooks

**Real-time Communication:**
- **WebSocket** - Browser <-> Go server
  - Endpoint: `/ws`
  - Purpose: Real-time video frame processing updates
  - Protocol: JSON-based frame protocol
  - Implementation: gorilla/websocket

- **gRPC Streaming** - Go gRPC client <-> C++ gRPC server
  - Service: `ImageProcessorService.StreamProcessVideo`
  - Purpose: Real-time video processing
  - Protocol: gRPC bidirectional streaming
  - Implementation: google.golang.org/grpc

- **MQTT** - Go server <-> IoT devices
  - Purpose: Power monitoring and device management
  - Topics: Sensor data, power commands
  - Implementation: Eclipse Paho MQTT

---

*Integration audit: 2026-04-12*
