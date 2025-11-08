#!/bin/bash
#
# Local Development Deployment
#
# This script builds and runs the application locally with:
# - CUDA-accelerated backend in Docker (built locally)
# - Traefik reverse proxy (HTTPS with auto redirect)
# - NVIDIA GPU passthrough
# - All services accessible via .localhost domains
#
# Usage:
#   ./scripts/deployment/local_dev/start.sh           # Build and run in background (default)
#   ./scripts/deployment/local_dev/start.sh --no-build # Skip build, just run
#   ./scripts/deployment/local_dev/start.sh --no-detach # Run in foreground (see logs)
#   ./scripts/deployment/local_dev/start.sh --help    # Show help message
#
# Note: Builds images locally using scripts/docker/build.sh
#       Uses local registry: local/base and local/intermediate
#
# Access:
#   https://app.localhost      - Main application
#   https://grafana.localhost  - Grafana
#   https://flipt.localhost    - Flipt
#   https://jaeger.localhost   - Jaeger
#   https://reports.localhost  - Test Reports
#
# Stop:
#   ./scripts/deployment/local_dev/stop.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

BUILD=true
DETACHED=true
SHOW_HELP=false
ARCH="amd64"

for arg in "$@"; do
    case "$arg" in
        --no-build)
            BUILD=false
            ;;
        --no-detach)
            DETACHED=false
            ;;
        --detach|-d)
            DETACHED=true
            ;;
        --arch)
            ARCH="$2"
            shift
            ;;
        --help|-h)
            SHOW_HELP=true
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: ./scripts/deployment/local_dev/start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-build            Skip building images, just run"
    echo "  --no-detach           Run in foreground (see logs)"
    echo "  --detach, -d           Run in background (default)"
    echo "  --arch ARCH           Architecture to build (amd64 or arm64) [default: amd64]"
    echo "  --help, -h             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/deployment/local_dev/start.sh           # Build + run in background"
    echo "  ./scripts/deployment/local_dev/start.sh --no-build # Just run (assumes images built)"
    echo "  ./scripts/deployment/local_dev/start.sh --no-detach # See logs"
    echo ""
    echo "Note: Builds images locally using scripts/docker/build.sh"
    echo "      Uses local registry: local/base and local/intermediate"
    echo ""
    echo "Stop:  ./scripts/deployment/local_dev/stop.sh"
    echo "Clean: ./scripts/deployment/local_dev/clean.sh"
    exit 0
fi

echo "Validating environment..."
echo ""
echo "[1/3] Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "  ERROR: Docker is not installed"
    exit 1
fi
echo "  OK: Docker is installed"

echo ""
echo "[2/3] Checking Docker daemon..."
if ! docker info &> /dev/null; then
    echo "  ERROR: Docker daemon is not running"
    exit 1
fi
echo "  OK: Docker daemon is running"

echo ""
echo "[3/3] Checking GPU availability..."
if docker run --rm --gpus all nvidia/cuda:12.5.1-runtime-ubuntu24.04 nvidia-smi &> /dev/null; then
    echo "  OK: GPU available"
    docker run --rm --gpus all nvidia/cuda:12.5.1-runtime-ubuntu24.04 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | head -1 | while read name driver memory; do
        echo "    $name, $driver, ${memory} MiB"
    done
else
    echo "  WARNING: GPU not available (CUDA may not be properly configured)"
fi

echo ""
echo "All validation checks passed"

if [ "$BUILD" = true ]; then
    echo ""
    echo "=========================================="
    echo "Building Docker images locally..."
    echo "=========================================="
    echo ""
    
    echo "Step 1: Building base and intermediate images..."
    "$PROJECT_ROOT/scripts/docker/build.sh" --arch "$ARCH"
    
    echo ""
    echo "Step 2: Building final application image..."
    
    PROTO_VERSION=$(cat "$PROJECT_ROOT/proto/VERSION" | tr -d '[:space:]')
    CPP_VERSION=$(cat "$PROJECT_ROOT/cpp_accelerator/VERSION" | tr -d '[:space:]')
    GOLANG_VERSION=$(cat "$PROJECT_ROOT/webserver/VERSION" | tr -d '[:space:]')
    
    export BASE_REGISTRY="local"
    export BASE_TAG="latest"
    export TARGETARCH="$ARCH"
    export PROTO_VERSION
    export CPP_VERSION
    export GOLANG_VERSION
    
    docker compose build app
    
    echo ""
    echo "  OK: All images built successfully"
fi

COMPOSE_CMD="docker compose -f docker-compose.yml up"
[ "$DETACHED" = true ] && COMPOSE_CMD="$COMPOSE_CMD -d"

echo ""
echo "Starting services..."
echo "Command: $COMPOSE_CMD"
echo ""

$COMPOSE_CMD

echo ""
echo "================================================"
echo "Local Development URLs (.localhost):"
echo "  App:     https://app.localhost"
echo "  Grafana: https://grafana.localhost"
echo "  Flipt:   https://flipt.localhost"
echo "  Jaeger:  https://jaeger.localhost"
echo "  Reports: https://reports.localhost"
echo "================================================"
echo ""
echo "Note: Browser will show security warning for self-signed certificates"
echo "      This is normal for local development environment"
echo ""

if [ "$DETACHED" = true ]; then
    echo "Services running in background"
    echo ""
    echo "View logs:"
    echo "  All services: docker compose -f docker-compose.yml logs -f"
    echo "  App only:     docker compose -f docker-compose.yml logs -f app"
    echo ""
    echo "Stop services: ./scripts/deployment/local_dev/stop.sh"
    echo "Clean up:     ./scripts/deployment/local_dev/clean.sh"
fi

