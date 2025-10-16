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
- [ ] Service discovery (Consul optional)
- [ ] Load balancing (round-robin, retry, circuit breaker)
- [x] Clean arch: domain → application → infrastructure → interfaces

## Observability

### Jaeger (Distributed Tracing)
- [x] Add Jaeger to docker-compose
- [x] OpenTelemetry SDK for Go
- [x] OpenTelemetry SDK for frontend (browser)
- [x] Distributed tracing from browser to backend
- [x] Trace context propagation with W3C headers
- [ ] Trace WebSocket, CGO calls, CUDA kernels
- [ ] Span attributes: width, height, filter type

### Prometheus (Metrics)
- [ ] Add Prometheus to docker-compose
- [ ] Expose /metrics endpoint
- [ ] Custom metrics: frames_processed, duration, connections
- [ ] GPU metrics exporter

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
- [ ] Add trace_id to logs
- [ ] Optional: Loki or ELK

## Load Testing & BDD

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
- [x] Feature: Input sources (5 scenarios)
- [x] Step definitions in Go
- [x] CI integration with dockerized tests
- [x] Cucumber HTML reports at port 5050

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

## Notes

Start with observability first. Keep it simple initially. GPU instances are expensive, monitor costs.

