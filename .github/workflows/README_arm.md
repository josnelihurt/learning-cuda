# ARM64 CI/CD Workflow

This document explains the ARM64 Docker build and deployment pipeline for the CUDA Learning Platform.

## Overview

The ARM64 workflow builds and deploys the `cpp-accelerator` image to Jetson Nano Orin devices. It implements smart CI with change detection to build only affected stages.

## File: `docker-monorepo-build-arm.yml`

### Triggers

| Event | Condition | Behavior |
|-------|-----------|----------|
| `push` | `main` branch | Build and push images |
| `pull_request` | `main` branch | Build only (no push) |
| `workflow_dispatch` | Manual | Full rebuild and deploy |

## Jobs

### 1. `detect-changes`

**Purpose**: Determine which components changed and what needs rebuilding.

**Outputs**:
| Output | Description |
|--------|-------------|
| `build_proto` | Build proto intermediates |
| `build_bazel_base` | Build bazel-base image |
| `build_cpp_deps` | Build cpp-dependencies image |
| `build_cuda_runtime` | Build cuda-runtime base image |
| `cpp_touched` | Any C++ related files changed |
| `cpp_version_changed` | `src/cpp_accelerator/VERSION` changed |

**Path Filters**:
```yaml
arm_proto:         proto/**, buf.yaml, buf.lock
arm_bazel_base:    src/cpp_accelerator/docker-build-base/**
arm_cpp_deps:      bazel/**, third_party/**, MODULE.bazel, WORKSPACE.bazel
arm_cuda_runtime:  src/cpp_accelerator/docker-cuda-runtime/**
arm_cpp_version:   src/cpp_accelerator/VERSION
arm_cpp_app:       src/cpp_accelerator/**/*.cpp, **/*.h, **/*.cu, **/BUILD
```

**Decision Logic**:
- `bazel-base` changes force `cpp-dependencies` rebuild
- `workflow_dispatch` forces full rebuild

---

### 2. `arm_pr`

**Purpose**: PR builds to validate changes without pushing.

**Conditions**: `github.event_name == 'pull_request' && cpp_touched == 'true'`

**Environment Variables**:
```bash
MODE=pr
BUILD_PROTO=0|1
BUILD_BAZEL_BASE=0|1
BUILD_CPP_DEPS=0|1
BUILD_CUDA_RUNTIME=0|1
```

**Script**: `scripts/ci/arm-build.sh`

**Stages Built** (in order):
1. `proto-tools` (if `BUILD_PROTO=1`)
2. `bazel-base` → `cpp-dependencies` (if `BUILD_BAZEL_BASE=1` or `BUILD_CPP_DEPS=1`)
3. `cuda-runtime` (if `BUILD_CUDA_RUNTIME=1`)
4. `cpp-builder` (always - compiles C++ from workspace HEAD)
5. Stops before `cpp-accelerator` (no runtime image in PRs)

---

### 3. `build_and_push`

**Purpose**: Build and push all images on merges to main.

**Conditions**: `(push || workflow_dispatch) && cpp_touched == 'true'`

**Environment Variables**:
```bash
MODE=push
BUILD_PROTO=0|1
BUILD_BAZEL_BASE=0|1
BUILD_CPP_DEPS=0|1
BUILD_CUDA_RUNTIME=0|1
```

**Script**: `scripts/ci/arm-build.sh`

**Stages Built** (all that changed):
1. All intermediates from `arm_pr`
2. `cpp-accelerator` (runtime image with C++ binary)

**Push Behavior**:
- Only pushes images that were rebuilt
- Skips pushing images pulled from GHCR cache

---

### 4. `deploy_prod`

**Purpose**: Deploy `cpp-accelerator` to Jetson Nano Orin.

**Conditions**:
```yaml
(push && main) || workflow_dispatch
&& cpp_version_changed == 'true'
```

**Key Behavior**: Only deploys when `src/cpp_accelerator/VERSION` is bumped.

**Deployment Steps**:
1. Validate Jetson SSH keys and secrets
2. Checkout repository
3. Configure SSH key
4. Add Jetson to known_hosts
5. Sync `docker-compose.yml` to Jetson
6. Sync mTLS certificates to Jetson
7. Pull new image via `docker compose`
8. Restart `cuda-accelerator-client` container

**Environment Variables** (from GitHub Secrets/Variables):
| Variable | Source | Description |
|----------|--------|-------------|
| `JETSON_HOST` | Secret | Jetson hostname/IP |
| `JETSON_USER` | Secret | SSH user |
| `JETSON_SSH_KEY` | Secret | SSH private key |
| `JETSON_APP_DIRECTORY` | Variable | Deployment directory on Jetson |
| `ACCELERATOR_CLIENT_CERT` | Secret | mTLS client certificate |
| `ACCELERATOR_CLIENT_KEY` | Secret | mTLS client key |
| `ACCELERATOR_CA_CERT` | Secret | mTLS CA certificate |

---

## Image Build Chain

```
nvidia/cuda:12.5.1-runtime-ubuntu24.04
    │
    ├─→ cuda-runtime (3.0.0) ──────────────────────────────┐
    │       │                                              │
    │       └─→ cpp-accelerator ───→ :latest-arm64        │
    │                                                      │
ubuntu:24.04                                             │
    │                                                     │
    ├─→ bazel-base ──→ cpp-dependencies ──→ cpp-builder ──┤
    │       │              │                 │            │
proto   ────────┘              │                 │            │
    │                          │                 │            │
    └─→ proto-generated ──────┴─────────────────┴────────────┘
```

---

## Script: `scripts/ci/arm-build.sh`

**Purpose**: Orchestrates ARM64 Docker builds based on change flags.

**Usage**:
```bash
MODE=pr|push
BUILD_PROTO=0|1
BUILD_BAZEL_BASE=0|1
BUILD_CPP_DEPS=0|1
BUILD_CUDA_RUNTIME=0|1
ARCH=arm64
REGISTRY=ghcr.io
BASE_IMAGE_PREFIX=josnelihurt-code/learning-cuda
```

**Build Logic**:
1. Build only stages whose inputs changed
2. Pull unchanged stages from GHCR
3. Always build `cpp-builder` (depends on workspace HEAD)
4. Push mode: push only rebuilt images

**Version Bump Detection**:
- Reads `src/cpp_accelerator/docker-*/VERSION` files
- Compares with GHCR to determine if push is needed

---

## Dependencies Between Files

```
docker-monorepo-build-arm.yml
    │
    ├─→ scripts/ci/arm-build.sh
    │       │
    │       ├─→ scripts/docker/build-local.sh
    │       │       │
    │       │       └─→ src/cpp_accelerator/docker-*/Dockerfile
    │       │       └─→ src/cpp_accelerator/Dockerfile.build
    │       │
    │       └─→ scripts/docker/pull-ghcr-cpp-intermediates.sh
    │
    └─→ src/cpp_accelerator/docker-compose.yml (deployed to Jetson)
```

---

## VERSION File Gating

| Component | VERSION File | Deploy Gate |
|-----------|--------------|-------------|
| cpp-accelerator | `src/cpp_accelerator/VERSION` | `cpp_version_changed == 'true'` |
| cuda-runtime | `src/cpp_accelerator/docker-cuda-runtime/VERSION` | Rebuilds image, no direct deploy gate |
| cpp-dependencies | `src/cpp_accelerator/docker-cpp-dependencies/VERSION` | Rebuilds image, no direct deploy gate |

**Why gate on VERSION?**
- Prevents unnecessary Jetson deployments
- Ensures production only gets versioned releases
- Allows config/docs changes without triggering deploys

---

## Architecture-Specific Details

### Jetson Nano Orin Target
- **Architecture**: ARM64 (aarch64)
- **JetPack**: 6 (R36)
- **CUDA**: 12.6
- **TensorRT**: 10.3

### Self-Hosted Runner
- **Runs on**: Jetson device or ARM64 machine
- **Tag**: `["self-hosted", "Linux", "ARM64", "dev"]`
- **Advantage**: Native compilation, no cross-compilation
