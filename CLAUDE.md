# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUDA Learning Platform - Real-time image/video processing via CUDA GPU kernels. Go web server communicates with C++/CUDA accelerator library via gRPC for distributed processing.

## Build & Development Commands

### Building Components
```bash
# C++ (Bazel) — requires TensorRT dev headers installed
bazel build //src/cpp_accelerator/...

# Go (from src/go_api/)
cd src/go_api && make build

# Frontend (from src/front-end/)
cd src/front-end && npm install && npm run dev
```

### Testing
```bash
# All coverage tests
./scripts/test/coverage.sh

# Unit tests
./scripts/test/unit-tests.sh
./scripts/test/unit-tests.sh --skip-golang   # Frontend only
./scripts/test/unit-tests.sh --skip-frontend # Go only

# Individual test suites
go test -race ./src/go_api/pkg/...                 # Go tests
cd src/front-end && npm run test                   # Frontend (Vitest)
bazel test //src/cpp_accelerator/...                # C++ tests

# Specific C++ test
bazel test //src/cpp_accelerator/core:logger_test

# BDD acceptance tests (requires services running)
go test ./test/integration/tests/acceptance -run TestFeatures -v

# E2E tests
./scripts/test/e2e.sh --chromium   # Fast: Chromium only
./scripts/test/e2e.sh              # All browsers
```

### Linting
```bash
./scripts/test/linters.sh          # All linters
./scripts/test/linters.sh --fix    # Auto-fix
```

## Architecture

### Code Structure
```
src/cpp_accelerator/
  application/         # Use cases, FilterPipeline, BufferPool
  domain/interfaces/   # IFilter, ImageBuffer, IImageProcessor
  infrastructure/cuda/ # CUDA kernel implementations
  infrastructure/cpu/  # CPU fallback implementations
  ports/grpc/         # gRPC service (primary integration)
  ports/shared_lib/   # Shared library exports
  core/               # Logger, Telemetry, Result type

src/go_api/
  cmd/server/         # main.go entry point
  pkg/app/            # Application bootstrap
  pkg/application/    # Use cases
  pkg/domain/         # Domain logic
  pkg/infrastructure/ # Repositories, gRPC client
  pkg/interfaces/     # HTTP/WebSocket handlers

src/front-end/        # React (Vite)
```


### Key Patterns
- Clean Architecture with dependency injection
- FilterPipeline orchestrates composable filter chains
- ProcessorEngine coordinates between ports and pipeline
- gRPC streaming for video processing (StreamProcessVideo)

## Testing Patterns

### Go Tests
- Use AAA comments (Arrange/Act/Assert)
- Use `testify/mock` (embed `mock.Mock`)
- Use `sut` for system under test
- Table-driven tests with `assertResult` function
- Naming: `Success_`, `Error_`, `Edge_` prefix
- Test data builders: `makeXXX()`

### TypeScript Tests (Vitest)
- Use AAA comments
- Use `vi.fn()` and `vi.mock()`
- Use `sut` for system under test
- Table-driven tests for multiple cases
- Naming: `Success_`, `Error_`, `Edge_` prefix
- Test data builders: `makeXXX()`

### C++ Tests (GoogleTest)
- Bazel test targets: `//src/cpp_accelerator/path:target_test`
- Equivalence tests verify CPU/CUDA produce identical results

## Configuration

- Development: `config/config.dev.yaml`
- Staging: `config/config.staging.yaml`
- Production: `config/config.production.yaml`

Proto generation: `./scripts/build/protos.sh`

## Tech Stack

- **Backend**: Go with native HTTPS, WebSocket support
- **Processing**: C++/CUDA via gRPC service (ConnectRPC)
- **Build**: Bazel for C++/CUDA, Makefile for Go
- **Frontend**: React + TypeScript with Vite
- **Observability**: OpenTelemetry, Jaeger tracing, Grafana dashboards, Loki logs

## TensorRT Setup (YOLO inference)

YOLO detection uses TensorRT. TensorRT dev headers must be installed — there is no stub fallback.

**Minimum supported version: TRT 10.0** — both target platforms run TRT 10.x:
- Jetson Nano Orin (JetPack 6 / R36): TRT 10.3, CUDA 12.6
- Dev PC (x86, RTX 4000): TRT 10.x, CUDA 12.5+

### x86 Ubuntu — install CUDA 12.9 + TensorRT dev headers

The TRT packages must match the CUDA version your driver supports.  
Driver 575.x supports **CUDA 12.9** — use the `+cuda12.9` TRT variant:

```bash
# If you accidentally have +cuda13.2 TRT (createInferBuilder error 35), run:
sudo bash scripts/dev/fix-cuda-trt-versions.sh

# Fresh install:
sudo apt-get install -y cuda-toolkit-12-9 \
  libnvinfer-dev=10.16.1.11-1+cuda12.9 \
  libnvonnxparsers-dev=10.16.1.11-1+cuda12.9
# Verify
dpkg -l | grep libnvinfer
```

### Build
```bash
bazel build //src/cpp_accelerator/...
# Or a single target
bazel build //src/cpp_accelerator/ports/shared_lib:libcuda_processor.so
```

### Dev stack
```bash
./scripts/dev/start.sh --build   # builds then starts all services
```

### Jetson Nano Orin / JetPack 6
TensorRT 10.x is pre-installed with JetPack 6. Build normally:
```bash
bazel build //src/cpp_accelerator/...
```
The code targets the TRT 10.x API exclusively: `getNbIOTensors`, `setTensorAddress`,
`enqueueV3`, `setMemoryPoolLimit`, and `buildSerializedNetwork`.

### ONNX → TRT engine caching
On first run with a new model, the detector builds a TRT engine from the `.onnx` file
and saves it as `.engine` (or `.jp6.engine` on Jetson Orin / aarch64). Subsequent runs
load the cached engine directly. The model path in `processor_engine.cpp` is:
`data/models/yolov10n.onnx`.

### Docker builds with TRT runtime
Pass `--build-arg ENABLE_TENSORRT=true` to include TRT runtime libs in the container:
```bash
docker build --build-arg ENABLE_TENSORRT=true -f src/cpp_accelerator/Dockerfile.build .
```

