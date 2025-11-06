#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARCH="amd64"
REGISTRY="local"
USE_COMPOSE=false
PUSH=false

display_help() {
    echo "Build the final Docker application image locally."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --arch ARCH          Architecture to build (amd64 or arm64) [default: amd64]"
    echo "  --registry REGISTRY  Registry prefix for images [default: local]"
    echo "  --compose            Use docker-compose to build (reads VERSION files)"
    echo "  --push               Push image to registry (requires --registry)"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --arch amd64                    # Build with docker build"
    echo "  $0 --arch amd64 --compose           # Build with docker-compose"
    echo "  $0 --arch amd64 --registry ghcr.io/josnelihurt/learning-cuda"
    echo ""
    echo "This script builds the final application image using:"
    echo "  - Base images (proto-tools, go-builder, bazel-base, etc.)"
    echo "  - Intermediate images (proto-generated, cpp-built, golang-built)"
    echo ""
    echo "VERSION files are read from:"
    echo "  - proto/VERSION"
    echo "  - cpp_accelerator/VERSION"
    echo "  - webserver/VERSION"
    echo ""
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --compose)
            USE_COMPOSE=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        -h|--help)
            display_help
            ;;
        *)
            echo "Unknown parameter passed: $1"
            display_help
            ;;
    esac
done

if [[ "$ARCH" != "amd64" && "$ARCH" != "arm64" ]]; then
    echo "Error: Invalid architecture '$ARCH'. Must be 'amd64' or 'arm64'."
    exit 1
fi

cd "$PROJECT_ROOT"

PROTO_VERSION=$(cat proto/VERSION | tr -d '[:space:]')
CPP_VERSION=$(cat cpp_accelerator/VERSION | tr -d '[:space:]')
GOLANG_VERSION=$(cat webserver/VERSION | tr -d '[:space:]')

echo "=========================================="
echo "Building Final Application Image"
echo "=========================================="
echo "Architecture: $ARCH"
echo "Registry: $REGISTRY"
echo "Proto version: $PROTO_VERSION"
echo "C++ version: $CPP_VERSION"
echo "Golang version: $GOLANG_VERSION"
echo ""

if [ "$USE_COMPOSE" = true ]; then
    echo "Using docker-compose to build..."
    echo ""
    
    export BASE_REGISTRY="$REGISTRY"
    export BASE_TAG="latest"
    export TARGETARCH="$ARCH"
    export PROTO_VERSION
    export CPP_VERSION
    export GOLANG_VERSION
    
    docker compose build app
    
    echo ""
    echo "=========================================="
    echo "Final image built successfully!"
    echo "=========================================="
    echo ""
    echo "Image: cuda-learning-app:latest"
    echo ""
    echo "To run with docker-compose:"
    echo "  docker compose up"
    echo ""
else
    echo "Using docker build directly..."
    echo ""
    
    IMAGE_TAG="${REGISTRY}/app:latest-${ARCH}"
    
    docker build \
        --platform "linux/${ARCH}" \
        --tag "$IMAGE_TAG" \
        --file Dockerfile \
        --build-arg "BASE_REGISTRY=${REGISTRY}" \
        --build-arg "BASE_TAG=latest" \
        --build-arg "TARGETARCH=${ARCH}" \
        --build-arg "PROTO_VERSION=${PROTO_VERSION}" \
        --build-arg "CPP_VERSION=${CPP_VERSION}" \
        --build-arg "GOLANG_VERSION=${GOLANG_VERSION}" \
        .
    
    if [ "$PUSH" = true ]; then
        echo ""
        echo "Pushing image to registry..."
        docker push "$IMAGE_TAG"
        echo "  OK: Image pushed to $IMAGE_TAG"
    fi
    
    echo ""
    echo "=========================================="
    echo "Final image built successfully!"
    echo "=========================================="
    echo ""
    echo "Image: $IMAGE_TAG"
    echo ""
    echo "To run:"
    echo "  docker run --gpus all -p 8080:8080 $IMAGE_TAG"
    echo ""
fi

