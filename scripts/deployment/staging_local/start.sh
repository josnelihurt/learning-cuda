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
#   ./scripts/deployment/staging_local/start.sh           # Run in background (default)
#   ./scripts/deployment/staging_local/start.sh --pull    # Pull latest image from GHCR
#   ./scripts/deployment/staging_local/start.sh --no-detach # Run in foreground (see logs)
#   ./scripts/deployment/staging_local/start.sh --help    # Show help message
#
# Note: Uses pre-built image from GitHub Container Registry (GHCR)
#       Image: ghcr.io/josnelihurt/learning-cuda:amd64-latest
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

PULL=false
DETACHED=true
SHOW_HELP=false

for arg in "$@"; do
    case "$arg" in
        --build|-b|--pull|-p)
            PULL=true
            ;;
        --no-detach)
            DETACHED=false
            ;;
        --detach|-d)
            DETACHED=true
            ;;
        --help|-h)
            SHOW_HELP=true
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: ./scripts/deployment/staging_local/start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build, -b, --pull, -p   Pull latest image from GHCR before starting"
    echo "  --no-detach                Run in foreground (see logs)"
    echo "  --detach, -d               Run in background (default)"
    echo "  --help, -h                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/deployment/staging_local/start.sh           # Background"
    echo "  ./scripts/deployment/staging_local/start.sh --pull    # Pull latest + start"
    echo "  ./scripts/deployment/staging_local/start.sh --no-detach # See logs"
    echo ""
    echo "Note: Uses pre-built image from GitHub Container Registry (GHCR)"
    echo "      Image: ghcr.io/josnelihurt/learning-cuda:amd64-latest"
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

if [ "$PULL" = true ]; then
    echo ""
    echo "Pulling latest image from GHCR..."
    docker pull ghcr.io/josnelihurt/learning-cuda:amd64-latest || {
        echo "  WARNING: Failed to pull image, using local cached version if available"
    }
    echo "  OK: Image pull completed"
fi

COMPOSE_CMD="docker compose -f docker-compose.staging.yml up"
[ "$DETACHED" = true ] && COMPOSE_CMD="$COMPOSE_CMD -d"

echo ""
echo "Starting staging services..."
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
    echo "  App only:     docker compose -f docker-compose.staging.yml logs -f app"
    echo ""
    echo "Stop services: ./scripts/deployment/staging_local/stop.sh"
    echo "Clean up:     ./scripts/deployment/staging_local/clean.sh"
fi
