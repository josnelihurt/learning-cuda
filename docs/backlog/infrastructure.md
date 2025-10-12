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
- [ ] Add Jaeger to docker-compose
- [ ] OpenTelemetry SDK for Go
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

### Logging
- [ ] spdlog (C++), zap/zerolog (Go)
- [ ] Add trace_id to logs
- [ ] Optional: Loki or ELK

## Load Testing & BDD

### K6 Load Tests
- [ ] Single user baseline (30 FPS for 30s)
- [ ] Concurrent users (ramp to 100, sustain, ramp down)
- [ ] Stress test (find breaking point)
- [ ] gRPC tests with `ghz`

### Godog BDD
- [ ] Setup Godog + feature files
- [ ] Feature: Video processing performance
- [ ] Feature: Filter pipeline
- [ ] Step definitions in Go
- [ ] CI integration, fail on regression

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

