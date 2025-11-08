#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARCH="amd64"
BUILD_BASE=true
BUILD_INTERMEDIATE=true

display_help() {
    echo "Build all Docker images locally (base + intermediate)."
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --arch ARCH          Architecture to build (amd64 or arm64) [default: amd64]"
    echo "  --skip-base          Skip building base images"
    echo "  --skip-intermediate  Skip building intermediate images"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --arch amd64                    # Build everything for amd64"
    echo "  $0 --skip-base                    # Only build intermediate images"
    echo "  $0 --skip-intermediate             # Only build base images"
    echo ""
    echo "This script builds:"
    echo "  1. Base images (proto-tools, go-builder, bazel-base, etc.)"
    echo "  2. Intermediate images (proto-generated, cpp-built, golang-built)"
    echo ""
    echo "After building, use build-final-image.sh to build the final application image."
    echo ""
    exit 0
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift
            ;;
        --skip-base)
            BUILD_BASE=false
            ;;
        --skip-intermediate)
            BUILD_INTERMEDIATE=false
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

echo "=========================================="
echo "Building Docker Images Locally"
echo "Architecture: $ARCH"
echo "=========================================="
echo ""

if [ "$BUILD_BASE" = true ]; then
    echo "=========================================="
    echo "Step 1/2: Building base images..."
    echo "=========================================="
    echo ""
    "$SCRIPT_DIR/build-local-base-images.sh" --arch "$ARCH"
    echo ""
fi

if [ "$BUILD_INTERMEDIATE" = true ]; then
    echo "=========================================="
    echo "Step 2/2: Building intermediate images..."
    echo "=========================================="
    echo ""
    "$SCRIPT_DIR/build-intermediate-images.sh" --arch "$ARCH"
    echo ""
fi

echo "=========================================="
echo "All images built successfully!"
echo "=========================================="
echo ""
echo "Next step: Build final application image"
echo "  ./scripts/docker/build-final-image.sh --arch $ARCH"
echo ""

