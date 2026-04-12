# Codebase Concerns

**Analysis Date:** 2026-04-12

## Tech Debt

**WebSocket to gRPC Migration (Incomplete):**
- Issue: Custom WebSocket protocol handling exists alongside newer gRPC streaming. TODO comments indicate gRPC bidirectional streaming should replace WebSocket, but migration is incomplete.
- Files: `webserver/pkg/interfaces/websocket/handler.go`, `webserver/pkg/interfaces/connectrpc/handler.go`, `webserver/web/src/infrastructure/transport/websocket-frame-transport.ts`
- Impact: Dual protocol maintenance, increased codebase complexity, potential protocol divergence
- Fix approach: Complete `StreamProcessVideo` implementation in `webserver/pkg/interfaces/connectrpc/handler.go`, update frontend to use Connect-Web streaming, deprecate WebSocket handler

**CGO Integration Path (Deprecated):**
- Issue: CGO shared library integration is deprecated but code still exists in codebase
- Files: `cpp_accelerator/ports/cgo/cgo_api.cpp`, `cpp_accelerator/ports/shared_lib/`
- Impact: Maintenance burden for unused code path, confusion about supported integration methods
- Fix approach: Remove CGO-related code once gRPC service is fully validated as replacement

**Video Player Stub Implementation:**
- Issue: `GoVideoPlayer.extractFrame()` is a stub that always returns "end of video" error, making the player non-functional
- Files: `webserver/pkg/infrastructure/video/go_video_player.go`
- Impact: Video playback feature cannot work with Go video player, forces dependency on FFmpeg player
- Fix approach: Implement actual frame extraction using FFmpeg or Go video library (github.com/u2takey/ffmpeg-go or github.com/giorgisio/goav)

**FFmpeg External Dependency:**
- Issue: Video processing relies on external FFmpeg binary via exec.Command rather than native Go library
- Files: `webserver/pkg/infrastructure/video/preview_generator.go`, `webserver/pkg/infrastructure/video/ffmpeg_video_player.go`
- Impact: External binary dependency, deployment complexity, limited error handling
- Fix approach: Replace exec.Command with Go library (github.com/u2takey/ffmpeg-go or github.com/giorgisio/goav)

**Frontend Singleton Pattern:**
- Issue: DIContainer uses singleton pattern with tight coupling, TODO indicates need for factory/builder pattern
- Files: `webserver/web/src/application/di/Container.ts`
- Impact: Difficult to test, creates global state, limits flexibility
- Fix approach: Implement dependency injection container with factory/builder pattern, remove singleton

## Known Bugs

**None documented.**

## Security Considerations

**InsecureSkipVerify in Development:**
- Risk: TLS certificate verification is disabled in development and test code
- Files: `integration/tests/acceptance/steps/bdd_context.go`, `webserver/pkg/interfaces/statichttp/development_handler.go`, `integration/tests/acceptance/bdd_helpers.go`, `integration/tests/acceptance/scripts/generate_checksums.go`
- Current mitigation: Used only in development/test environments, documented in testing docs
- Recommendations: Ensure development-specific code paths never reach production, add runtime checks to prevent InsecureSkipVerify in production builds

**Secrets Management:**
- Risk: Secrets directory exists with certificates and credentials, gitignore properly configured but manual verification needed
- Files: `.secrets/` directory structure
- Current mitigation: `.gitignore` properly excludes `.pem`, `.key`, `.crt`, `.env` files, secrets in `.secrets/` directory
- Recommendations: Regular audits of `.secrets/` directory, consider secret scanning tools, rotate certificates periodically

**External Process Execution:**
- Risk: FFmpeg executed via exec.Command with user-controlled paths (though validated by caller)
- Files: `webserver/pkg/infrastructure/video/ffmpeg_video_player.go`, `webserver/pkg/infrastructure/video/preview_generator.go`
- Current mitigation: #nosec G204 comments indicate paths are validated by caller, context with timeout for cancellation
- Recommendations: Add explicit path validation whitelist, consider sandboxing FFmpeg processes

**WebSocket Origin Check:**
- Risk: WebSocket upgrader accepts all origins (CheckOrigin returns true)
- Files: `webserver/pkg/interfaces/websocket/handler.go`
- Current mitigation: Used in development with TLS, production deployments should have proper origin validation
- Recommendations: Implement proper origin validation for production, whitelist allowed domains

## Performance Bottlenecks

**Large Generated Files:**
- Problem: Generated protobuf files are very large (2424 lines for image_processor_service.pb.go, 1221 lines for image_processor_service_pb.ts)
- Files: `proto/gen/image_processor_service.pb.go`, `webserver/web/src/gen/image_processor_service_pb.ts`
- Cause: Large protobuf definitions with many message types
- Improvement path: Consider breaking down large services into smaller focused services, but this may not be worth the effort

**Time.Sleep in Tests:**
- Problem: Tests use hardcoded time.Sleep() for synchronization, making tests slow and flaky
- Files: `integration/tests/acceptance/steps/bdd_context.go`, `webserver/pkg/infrastructure/logger/logger_integration_test.go`, `integration/tests/acceptance/steps/frontend_logging_steps.go`
- Cause: Waiting for async operations to complete
- Improvement path: Replace with proper synchronization primitives (channels, wait groups, context cancellation)

**BDD Test Context Size:**
- Problem: BDD context file is 2424 lines with many responsibilities
- Files: `integration/tests/acceptance/steps/bdd_context.go`
- Cause: God object pattern accumulating test state and helpers
- Improvement path: Split into smaller focused test helpers by feature area

**Video Processing Latency:**
- Problem: Video playback processes frames sequentially without backpressure management
- Files: `webserver/pkg/interfaces/websocket/handler.go` (streamRealVideo method)
- Cause: Sequential frame processing in single goroutine
- Improvement path: Implement frame buffering and backpressure, consider parallel frame processing with worker pool

## Fragile Areas

**WebSocket Handler Complexity:**
- Files: `webserver/pkg/interfaces/websocket/handler.go` (562 lines)
- Why fragile: Complex goroutine management with connection tracking, mutex maps, video session management, custom protocol handling
- Safe modification: Add comprehensive tests for goroutine leaks, race conditions; use race detector in tests; consider extracting connection management to separate module
- Test coverage: Limited for concurrent scenarios, needs stress testing

**BDD Test Suite:**
- Files: `integration/tests/acceptance/steps/bdd_context.go`, `integration/tests/acceptance/steps/then_steps.go`
- Why fragile: Depends on external services (Flipt, gRPC server), uses time.Sleep for synchronization, large shared context state
- Safe modification: Fixtures should be isolated, use proper synchronization instead of Sleep, mock external dependencies where possible
- Test coverage: Acceptance tests are integration-level, not unit tests; dependencies on running services

**C++ Resource Management:**
- Files: `cpp_accelerator/ports/shared_lib/processor_api.h`
- Why fragile: Manual memory management with explicit cleanup requirements, not thread-safe for init/cleanup
- Safe modification: Follow documented threading constraints, always call processor_free_response() for non-NULL responses, ensure single-threaded init/cleanup
- Test coverage: Resource cleanup tests exist but may not cover all failure paths

**Feature Flags Synchronization:**
- Files: `webserver/pkg/infrastructure/featureflags/flipt_writer.go`
- Why fragile: Manual HTTP-based synchronization with Flipt, depends on external service availability
- Safe modification: Add circuit breaker pattern, implement retry logic with exponential backoff, add health checks
- Test coverage: Integration tests cover happy path, limited failure scenario testing

## Scaling Limits

**Single-Server Architecture:**
- Current capacity: Single Go server instance, single gRPC C++ accelerator service
- Limit: No horizontal scaling, session state in WebSocket handler is in-memory
- Scaling path: Extract session state to Redis, implement load balancer with sticky sessions or replace WebSocket with gRPC streaming

**Video Processing Throughput:**
- Current capacity: Sequential frame processing, limited by single GPU
- Limit: Video playback processes frames one at a time, no parallel processing
- Scaling path: Implement frame-level parallelism with worker pool, support multiple GPU instances, distribute processing across multiple accelerator services

**Bazel Build Times:**
- Current capacity: Bazel builds C++/CUDA code from scratch
- Limit: Full rebuilds can take significant time, especially with CUDA compilation
- Scaling path: Use remote build cache, implement incremental builds, consider pre-built docker images for dependencies

## Dependencies at Risk

**rules_cuda (Bazel module):**
- Risk: Third-party Bazel module for CUDA support, requires archive override
- Impact: CUDA build tooling breaks if rules_cuda compatibility changes
- Migration plan: Monitor rules_cuda releases, test with new Bazel versions, consider contributing to BCR (Bazel Central Registry)

**Gorilla WebSocket:**
- Risk: Custom WebSocket protocol implementation, will need migration away from WebSocket
- Impact: Maintenance burden until gRPC streaming migration complete
- Migration plan: Complete gRPC streaming implementation, deprecate WebSocket handler, remove dependency

**FFmpeg Binary:**
- Risk: External binary dependency, version compatibility issues
- Impact: Video processing fails if FFmpeg not installed or version incompatible
- Migration plan: Replace with Go library (github.com/u2takey/ffmpeg-go) as noted in TODO comments

**Connect-RPC Vanguard:**
- Risk: Relatively new transcoder library (v0.3.0), rapid development
- Impact: API changes or bugs in Vanguard could break service routing
- Migration plan: Pin to specific versions, test thoroughly before upgrades, monitor project activity

## Missing Critical Features

**gRPC Bidirectional Streaming:**
- Problem: StreamProcessVideo returns CodeUnimplemented, video streaming still uses WebSocket
- Blocks: Complete migration to type-safe streaming protocol, unified API surface
- Status: TODO indicates planned implementation, handler stub exists

**Comprehensive Error Handling:**
- Problem: Some error paths return generic errors without context
- Blocks: Debugging production issues, proper error recovery
- Status: Error handling exists but inconsistent across codebase

**Backpressure Management:**
- Problem: No explicit backpressure handling in streaming video, potential for resource exhaustion
- Blocks: Reliable video streaming under load, preventing OOM
- Status: Not implemented, relies on implicit flow control

## Test Coverage Gaps

**Concurrent Code Testing:**
- What's not tested: Goroutine leaks, race conditions in WebSocket handler, concurrent access to shared state
- Files: `webserver/pkg/interfaces/websocket/handler.go`, `webserver/pkg/interfaces/connectrpc/remote_management_handler.go`, `webserver/pkg/infrastructure/mqtt/device_monitor.go`
- Risk: Race conditions in production, resource leaks, deadlocks
- Priority: High

**Error Path Testing:**
- What's not tested: Failure scenarios in video processing, network errors in gRPC calls, resource exhaustion
- Files: `webserver/pkg/application/video_playback_use_case.go`, `webserver/pkg/interfaces/connectrpc/handler.go`
- Risk: Poor error recovery, cascading failures
- Priority: Medium

**C++ Resource Cleanup:**
- What's not tested: All failure paths for CUDA memory allocation, cleanup on errors
- Files: `cpp_accelerator/ports/shared_lib/processor_api.h`, CUDA kernel implementations
- Risk: Memory leaks in C++ accelerator service, GPU memory exhaustion
- Priority: High

**Integration Test Flakiness:**
- What's not tested: Proper synchronization without time.Sleep, independent test isolation
- Files: `integration/tests/acceptance/steps/bdd_context.go`, BDD test suite
- Risk: Flaky tests, false negatives, slowed CI/CD
- Priority: Medium

---

*Concerns audit: 2026-04-12*
