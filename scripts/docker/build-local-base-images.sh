#!/bin/bash

#################################################################################
# LESSON LEARNT: Buildx Cache with HTTP Registry
#################################################################################
# 
# Initially, we attempted to use a local Docker registry (HTTP) for Buildx cache
# to persist cache between builds. However, this approach had fundamental issues:
#
# 1. BuildKit (used by Buildx) runs in a container and attempts to use HTTPS
#    by default when accessing registries, even when configured as insecure.
#
# 2. Although we added 172.17.0.1:5000 to insecure-registries in daemon.json,
#    BuildKit inside the container doesn't automatically respect these settings.
#
# 3. We tried multiple approaches:
#    - Using localhost:5000 (doesn't work from container)
#    - Using Docker bridge IP 172.17.0.1:5000 (HTTPS/HTTP mismatch)
#    - Configuring BuildKit with buildkitd.toml (not accessible from container)
#    - Using network=host driver option (still uses HTTPS)
#
# 4. Result: Buildx cache export/import to HTTP registry fails with:
#    "http: server gave HTTP response to HTTPS client"
#
# Solution: Use docker build (default) for local builds - it has excellent
# daemon cache support. Use Buildx only when multi-arch is needed, accepting
# that cache will be internal to Buildx and not persisted to registry.
#
#################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

REGISTRY="local"
BASE_IMAGE_PREFIX="base"
ARCH="amd64"
PLATFORM="linux/amd64"

show_help() {
    cat <<EOF
Build all Docker base images locally for local development.

Usage: $0 [OPTIONS]

Options:
  --arch ARCH          Architecture to build (amd64 or arm64) [default: amd64]
  --buildx             Use Docker Buildx instead of docker build (requires registry setup)
  --help, -h           Show this help message

Examples:
  $0 --arch amd64
  $0 --arch arm64

This script builds the following base images:
  - proto-tools (from proto/Dockerfile)
  - go-builder (from webserver/builder/Dockerfile)
  - integration-tests-base (from integration/Dockerfile)
  - runtime-base (from runtime/Dockerfile)
  - bazel-base (from cpp_accelerator/docker-build-base/Dockerfile)

Images are tagged as: ${REGISTRY}/${BASE_IMAGE_PREFIX}:{image-name}-latest-{arch}

After building, use them with docker-compose:
  BASE_REGISTRY=${REGISTRY} docker-compose build
EOF
}

USE_BUILDX_FLAG=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --buildx)
            USE_BUILDX_FLAG=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ "$ARCH" != "amd64" ] && [ "$ARCH" != "arm64" ]; then
    echo "ERROR: Invalid architecture '$ARCH'. Must be 'amd64' or 'arm64'"
    exit 1
fi

if [ "$ARCH" = "arm64" ]; then
    PLATFORM="linux/arm64"
fi

echo "Building Docker base images locally"
echo "Registry: ${REGISTRY}"
echo "Architecture: ${ARCH}"
echo "Platform: ${PLATFORM}"
echo ""

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running"
    exit 1
fi

# Default to docker build for local builds (better cache support)
# Use --buildx flag to enable Buildx (for multi-arch builds)
# Note: Buildx uses internal cache only (not persisted to registry due to HTTP/HTTPS issues)
USE_BUILDX=false
if [ "$USE_BUILDX_FLAG" = true ]; then
    if command -v docker &> /dev/null && docker buildx version &> /dev/null 2>&1; then
        USE_BUILDX=true
        echo "Using Docker Buildx for builds"
        echo "Note: Buildx will use internal cache (not persisted to registry)"
    else
        echo "ERROR: --buildx flag specified but Buildx is not available"
        exit 1
    fi
else
    echo "Using Docker build (daemon cache - recommended for local builds)"
fi
echo ""

build_image() {
    local name=$1
    local dockerfile=$2
    local version_tag=$3
    local latest_tag="${REGISTRY}/${BASE_IMAGE_PREFIX}:${name}-latest-${ARCH}"
    local versioned_tag="${REGISTRY}/${BASE_IMAGE_PREFIX}:${name}-${version_tag}-${ARCH}"
    
    echo "=========================================="
    echo "Building: ${name}"
    echo "Dockerfile: ${dockerfile}"
    echo "Tags: ${latest_tag}, ${versioned_tag}"
    echo "=========================================="
    
    # Build the command as an array to display and execute it
    local cmd_args=()
    if [ "$USE_BUILDX" = true ]; then
        # Use Buildx for multi-arch builds
        # Note: Cache is internal to Buildx (not persisted to registry)
        # See lesson learnt comment at the top of this script for details
        cmd_args=(
            "docker" "buildx" "build"
            "--platform" "${PLATFORM}"
            "--build-arg" "TARGETARCH=${ARCH}"
            "-f" "${dockerfile}"
            "-t" "${latest_tag}"
            "-t" "${versioned_tag}"
            "--load"
            "."
        )
    else
        cmd_args=(
            "docker" "build"
            "--build-arg" "TARGETARCH=${ARCH}"
            "-f" "${dockerfile}"
            "-t" "${latest_tag}"
            "-t" "${versioned_tag}"
            "."
        )
    fi
    
    echo ""
    echo ">>> Executing command:"
    # Build command string for display (properly quoted)
    local cmd_display=""
    for arg in "${cmd_args[@]}"; do
        if [[ "$arg" =~ [[:space:]] ]]; then
            cmd_display="${cmd_display} \"${arg}\""
        else
            cmd_display="${cmd_display} ${arg}"
        fi
    done
    echo ">>>${cmd_display}"
    echo ""
    
    # Execute the command
    "${cmd_args[@]}"
    
    echo ">>> Verifying image: docker image inspect ${latest_tag}"
    if ! docker image inspect "${latest_tag}" &> /dev/null; then
        echo "ERROR: Failed to verify image ${latest_tag}"
        exit 1
    fi
    echo ">>> Image verified successfully"
    
    local localhost_latest_tag="localhost/${BASE_IMAGE_PREFIX}:${name}-latest-${ARCH}"
    local localhost_versioned_tag="localhost/${BASE_IMAGE_PREFIX}:${name}-${version_tag}-${ARCH}"
    
    echo ">>> Tagging as localhost/ for local Docker recognition..."
    docker tag "${latest_tag}" "${localhost_latest_tag}" || true
    docker tag "${versioned_tag}" "${localhost_versioned_tag}" || true
    
    echo ""
    echo "OK: Successfully built ${name}"
    echo ""
}

echo "Starting base images build..."
echo ""

build_image "proto-tools" "proto/Dockerfile" "v1.47.2"
build_image "go-builder" "webserver/builder/Dockerfile" "1.24"
build_image "integration-tests-base" "integration/Dockerfile" "1.24.0"
build_image "runtime-base" "runtime/Dockerfile" "12.5.1"
NVIDIA_BASE_IMAGE="nvidia/cuda:12.5.1-devel-ubuntu24.04"
docker pull "$NVIDIA_BASE_IMAGE"
build_image "bazel-base" "cpp_accelerator/docker-build-base/Dockerfile" "7.0.2"

echo "=========================================="
echo "All base images built successfully!"
echo "=========================================="
echo ""
echo ">>> Listing images: docker images | grep \"^${REGISTRY}/${BASE_IMAGE_PREFIX}\""
docker images | grep "^${REGISTRY}/${BASE_IMAGE_PREFIX}" || true
echo ""
echo "To use these images with docker-compose:"
echo "  BASE_REGISTRY=${REGISTRY} docker-compose build"
echo ""

