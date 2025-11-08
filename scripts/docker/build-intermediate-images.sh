#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARCH="amd64"
REGISTRY="local"
BASE_IMAGE_PREFIX="base"

display_help() {
    echo "Build all Docker intermediate images locally for local development."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --arch ARCH          Architecture to build (amd64 or arm64) [default: amd64]"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --arch amd64"
    echo "  $0 --arch arm64"
    echo ""
    echo "This script builds the following intermediate images:"
    echo "  - proto-generated (from proto/Dockerfile.generated)"
    echo "  - cpp-built (from cpp_accelerator/Dockerfile.build)"
    echo "  - golang-built (from webserver/Dockerfile.build)"
    echo ""
    echo "Images are tagged as: ${REGISTRY}/${BASE_IMAGE_PREFIX}/intermediate:{name}-{version}-{arch}"
    echo ""
    echo "After building, use them with docker-compose:"
    echo "  BASE_REGISTRY=${REGISTRY} PROTO_VERSION=\$(cat proto/VERSION) CPP_VERSION=\$(cat cpp_accelerator/VERSION) GOLANG_VERSION=\$(cat webserver/VERSION) docker-compose build"
    echo ""
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
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
    shift
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
echo "Building local intermediate images for ${ARCH}..."
echo "=========================================="
echo "Proto version: $PROTO_VERSION"
echo "C++ version: $CPP_VERSION"
echo "Golang version: $GOLANG_VERSION"
echo ""

echo "Using standard Docker build for local images..."
echo ""

build_image() {
    local name=$1
    local dockerfile_path=$2
    local version=$3
    local context_path=${4:-.}
    local build_args=${5:-""}

    local versioned_tag="${REGISTRY}/intermediate:${name}-${version}-${ARCH}"
    local latest_tag="${REGISTRY}/intermediate:${name}-latest-${ARCH}"

    echo "=========================================="
    echo "Building: ${name}"
    echo "Dockerfile: ${dockerfile_path}"
    echo "Tags: ${versioned_tag}, ${latest_tag}"
    echo "=========================================="

    local build_cmd="docker build \
        --platform \"linux/${ARCH}\" \
        --tag \"${versioned_tag}\" \
        --tag \"${latest_tag}\" \
        --file \"${dockerfile_path}\" \
        --build-arg \"TARGETARCH=${ARCH}\" \
        --build-arg \"BASE_REGISTRY=${REGISTRY}\" \
        --build-arg \"BASE_TAG=latest\""

    if [ -n "$build_args" ]; then
        build_cmd="${build_cmd} ${build_args}"
    fi

    build_cmd="${build_cmd} \"${context_path}\""

    eval $build_cmd || {
        echo "ERROR: Failed to build ${name} image."
        exit 1
    }
    
    local localhost_versioned_tag="localhost/intermediate:${name}-${version}-${ARCH}"
    local localhost_latest_tag="localhost/intermediate:${name}-latest-${ARCH}"
    
    echo ">>> Tagging as localhost/ for local Docker recognition..."
    docker tag "${versioned_tag}" "${localhost_versioned_tag}" || true
    docker tag "${latest_tag}" "${localhost_latest_tag}" || true
    
    echo "[OK] Successfully built ${name}"
    echo ""
}

build_image "proto-generated" "proto/Dockerfile.generated" "$PROTO_VERSION"

build_image "cpp-built" "cpp_accelerator/Dockerfile.build" "$CPP_VERSION" "." "--build-arg PROTO_VERSION=${PROTO_VERSION}"

build_image "golang-built" "webserver/Dockerfile.build" "$GOLANG_VERSION" "." "--build-arg PROTO_VERSION=${PROTO_VERSION}"

echo "=========================================="
echo "All intermediate images built successfully!"
echo "=========================================="
echo ""
echo "Images available:"
docker images | grep "^${REGISTRY}/${BASE_IMAGE_PREFIX}/intermediate" || true
echo ""
echo "To use these images with docker-compose:"
echo "  BASE_REGISTRY=${REGISTRY} \\"
echo "  PROTO_VERSION=${PROTO_VERSION} \\"
echo "  CPP_VERSION=${CPP_VERSION} \\"
echo "  GOLANG_VERSION=${GOLANG_VERSION} \\"
echo "  docker-compose build"
echo ""

