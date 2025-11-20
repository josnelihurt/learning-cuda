#!/bin/bash
#
# Staging Local Deployment
#
# This script runs the staging stack locally with:
# - CUDA-accelerated backend in Docker
# - Traefik reverse proxy (HTTPS with auto redirect)
# - NVIDIA GPU passthrough
# - All services accessible via .localhost domains
#
# Usage:
#   ./scripts/deployment/staging_local/start.sh                    # Run with GHCR images (default)
#   ./scripts/deployment/staging_local/start.sh --registry local  # Build and use local images
#   ./scripts/deployment/staging_local/start.sh --pull            # Pull latest image from GHCR
#   ./scripts/deployment/staging_local/start.sh --no-detach        # Run in foreground (see logs)
#   ./scripts/deployment/staging_local/start.sh --help            # Show help message
#
# Registry options:
#   - ghcr.io/josnelihurt/learning-cuda (default) - Use pre-built images from GHCR
#   - local                                    - Build images locally before running
#
# Access:
#   https://app.localhost      - Main application
#   https://grafana.localhost  - Grafana
#   https://flipt.localhost    - Flipt
#   https://jaeger.localhost   - Jaeger
#   https://reports.localhost  - Test Reports
#
# Stop:
#   ./scripts/deployment/staging_local/stop.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

REGISTRY="ghcr.io/josnelihurt/learning-cuda"
PULL=false
BUILD=false
DETACHED=true
SHOW_HELP=false
ARCH="amd64"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            if [ "$REGISTRY" = "local" ]; then
                BUILD=true
            fi
            shift 2
            ;;
        --build|-b|--pull|-p)
            PULL=true
            shift
            ;;
        --no-build)
            BUILD=false
            shift
            ;;
        --no-detach)
            DETACHED=false
            shift
            ;;
        --detach|-d)
            DETACHED=true
            shift
            ;;
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: ./scripts/deployment/staging_local/start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --registry REGISTRY       Registry to use [default: ghcr.io/josnelihurt/learning-cuda]"
    echo "                            Use 'local' to build images locally"
    echo "  --build, -b, --pull, -p    Pull latest image from GHCR before starting"
    echo "  --no-build                Skip building when using --registry local"
    echo "  --no-detach               Run in foreground (see logs)"
    echo "  --detach, -d              Run in background (default)"
    echo "  --arch ARCH               Architecture to build (amd64 or arm64) [default: amd64]"
    echo "  --help, -h                Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/deployment/staging_local/start.sh                    # Use GHCR images"
    echo "  ./scripts/deployment/staging_local/start.sh --registry local   # Build locally + run"
    echo "  ./scripts/deployment/staging_local/start.sh --pull             # Pull latest from GHCR"
    echo "  ./scripts/deployment/staging_local/start.sh --no-detach        # See logs"
    echo ""
    echo "Registry options:"
    echo "  - ghcr.io/josnelihurt/learning-cuda (default) - Pre-built images from GHCR"
    echo "  - local                                    - Build images locally"
    echo ""
    echo "Stop:  ./scripts/deployment/staging_local/stop.sh"
    echo "Clean: ./scripts/deployment/staging_local/clean.sh"
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
echo "[2/2] Checking Docker daemon..."
if ! docker info &> /dev/null; then
    echo "  ERROR: Docker daemon is not running"
    exit 1
fi
echo "  OK: Docker daemon is running"

echo ""
echo "All validation checks passed"

if [ "$BUILD" = true ]; then
    echo ""
    echo "=========================================="
    echo "Building Docker images locally..."
    echo "=========================================="
    echo ""
    
    echo "Step 1: Building base and intermediate images (including gRPC server)..."
    "$PROJECT_ROOT/scripts/docker/build-local.sh" --arch "$ARCH" --registry local --stage proto-tools --stage go-builder --stage bazel-base --stage runtime-base --stage integration-base --stage proto --stage cpp --stage golang --stage grpc-server
    
    echo ""
    echo "Step 2: Building final application image..."
    
    PROTO_VERSION=$(cat "$PROJECT_ROOT/proto/VERSION" | tr -d '[:space:]')
    CPP_VERSION=$(cat "$PROJECT_ROOT/cpp_accelerator/VERSION" | tr -d '[:space:]')
    GOLANG_VERSION=$(cat "$PROJECT_ROOT/webserver/VERSION" | tr -d '[:space:]')
    
    export BASE_REGISTRY="local/josnelihurt/learning-cuda"
    export BASE_TAG="latest"
    export TARGETARCH="$ARCH"
    export PROTO_VERSION
    export CPP_VERSION
    export GOLANG_VERSION
    
    DOCKER_BUILDKIT=1 docker build \
        --platform "linux/${ARCH}" \
        --tag "cuda-learning-app:latest" \
        --file Dockerfile \
        --build-arg "BASE_REGISTRY=${BASE_REGISTRY}" \
        --build-arg "BASE_TAG=${BASE_TAG}" \
        --build-arg "TARGETARCH=${TARGETARCH}" \
        --build-arg "PROTO_VERSION=${PROTO_VERSION}" \
        --build-arg "CPP_VERSION=${CPP_VERSION}" \
        --build-arg "GOLANG_VERSION=${GOLANG_VERSION}" \
        .
    
    echo ""
    echo "  OK: All images built successfully"
elif [ "$PULL" = true ]; then
    echo ""
    echo "Pulling latest image from GHCR..."
    docker pull ghcr.io/josnelihurt/learning-cuda:amd64-latest || {
        echo "  WARNING: Failed to pull image, using local cached version if available"
    }
    echo "  OK: Image pull completed"
fi

echo ""
echo "Removing app container if exists..."
docker rm -f cuda-image-processor-staging 2>/dev/null || true

if [ "$REGISTRY" = "local" ]; then
    export APP_IMAGE="cuda-learning-app:latest"
    export GRPC_SERVER_IMAGE="local/josnelihurt/learning-cuda/grpc-server:latest-${ARCH}"
    echo "Using local images:"
    echo "  App:        $APP_IMAGE"
    echo "  gRPC Server: $GRPC_SERVER_IMAGE"
else
    export APP_IMAGE="ghcr.io/josnelihurt/learning-cuda:amd64-latest"
    export GRPC_SERVER_IMAGE="ghcr.io/josnelihurt/learning-cuda/grpc-server:latest-amd64"
    echo "Using GHCR images:"
    echo "  App:        $APP_IMAGE"
    echo "  gRPC Server: $GRPC_SERVER_IMAGE"
fi

COMPOSE_CMD="docker compose -f docker-compose.staging.yml up"
[ "$DETACHED" = true ] && COMPOSE_CMD="$COMPOSE_CMD -d"

echo ""
echo "Starting services..."
echo "Command: $COMPOSE_CMD"
echo ""

$COMPOSE_CMD

echo ""
echo "================================================"
echo "Staging Local URLs (.localhost):"
echo "  App:     https://app.localhost"
echo "  Grafana: https://grafana.localhost"
echo "  Flipt:   https://flipt.localhost"
echo "  Jaeger:  https://jaeger.localhost"
echo "  Reports: https://reports.localhost"
echo "================================================"
echo ""
echo "Note: Browser will show security warning for self-signed certificates"
echo "      This is normal for local staging environment"
echo ""

if [ "$DETACHED" = true ]; then
    echo "Services running in background"
    echo ""
    echo "View logs:"
    echo "  All services: docker compose -f docker-compose.staging.yml logs -f"
    echo ""
    echo "Stop services: ./scripts/deployment/staging_local/stop.sh"
    echo "Clean up:     ./scripts/deployment/staging_local/clean.sh"
fi
