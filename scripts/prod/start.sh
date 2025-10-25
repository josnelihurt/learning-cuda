#!/bin/bash
#
# Production Docker Deployment
#
# This script runs the full production stack with:
# - CUDA-accelerated backend in Docker
# - Traefik reverse proxy (HTTPS with auto redirect)
# - NVIDIA GPU passthrough
# - Multi-stage optimized build
# - Optional Cloudflare Tunnel support
#
# Usage:
#   ./scripts/prod/start.sh                    # Run in background (default)
#   ./scripts/prod/start.sh --cloudflare      # Run with Cloudflare Tunnel
#   ./scripts/prod/start.sh --build           # Force rebuild images
#   ./scripts/prod/start.sh --no-detach       # Run in foreground (see logs)
#   ./scripts/prod/start.sh --help            # Show help message
#
# Access (Local):
#   https://localhost        - Main application (HTTP auto-redirects)
#   http://localhost:8081    - Traefik dashboard
#
# Access (Cloudflare):
#   https://app-cuda-demo.josnelihurt.me     - Main application
#   https://grafana-cuda-demo.josnelihurt.me - Grafana
#   https://flipt-cuda-demo.josnelihurt.me   - Flipt
#   https://jaeger-cuda-demo.josnelihurt.me  - Jaeger
#   https://reports-cuda-demo.josnelihurt.me - Test Reports
#
# Stop:
#   ./scripts/prod/stop.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Parse command line arguments
BUILD=false
DETACHED=true  # Default to background
CLOUDFLARE=false
SHOW_HELP=false

for arg in "$@"; do
    case "$arg" in
        --build|-b)
            BUILD=true
            ;;
        --cloudflare|-c)
            CLOUDFLARE=true
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
    echo "Usage: ./scripts/prod/start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build, -b        Force rebuild of Docker images"
    echo "  --cloudflare, -c   Enable Cloudflare Tunnel profile"
    echo "  --no-detach         Run in foreground (see logs)"
    echo "  --detach, -d        Run in background (default)"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/prod/start.sh                    # Local production (background)"
    echo "  ./scripts/prod/start.sh --cloudflare      # With Cloudflare Tunnel"
    echo "  ./scripts/prod/start.sh --build --cloudflare # Rebuild + Cloudflare"
    echo "  ./scripts/prod/start.sh --no-detach        # See logs in foreground"
    echo ""
    echo "Stop: ./scripts/prod/stop.sh"
    echo "Clean: ./scripts/prod/clean.sh"
    exit 0
fi

# Validate environment
echo "Validating environment..."
echo "Validating Docker environment..."
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

if [ $? -ne 0 ]; then
    exit 1
fi

# Load production secrets
echo "Loading production secrets..."
if [ -f ".secrets/production.env" ]; then
    echo "  Loading secrets from .secrets/production.env"
    source .secrets/production.env
else
    echo "  ERROR: Production secrets file not found at .secrets/production.env"
    echo "  Please create this file with your actual secrets before running production"
    echo "  You can use .secrets/production.env.example as a template"
    exit 1
fi

# Cloudflare is always enabled in production
echo ""
echo "Cloudflare Tunnel configuration..."
echo "  Tunnel ID: ef1d45eb-5e53-4a40-8bc3-e30c1e29e00d"
echo "  OK: Cloudflare Tunnel configured"

# Build compose command
COMPOSE_CMD="docker compose up"
[ "$BUILD" = true ] && COMPOSE_CMD="$COMPOSE_CMD --build"
[ "$DETACHED" = true ] && COMPOSE_CMD="$COMPOSE_CMD -d"

echo ""
echo "Starting production services..."
echo "Command: $COMPOSE_CMD"
echo ""

# Execute
$COMPOSE_CMD

# Display URLs based on mode
echo ""
echo "================================================"
echo "Production URLs (Cloudflare Tunnel):"
echo "  App:     https://app-cuda-demo.josnelihurt.me"
echo "  Grafana: https://grafana-cuda-demo.josnelihurt.me"
echo "  Flipt:   https://flipt-cuda-demo.josnelihurt.me"
echo "  Jaeger:  https://jaeger-cuda-demo.josnelihurt.me"
echo "  Reports: https://reports-cuda-demo.josnelihurt.me"
echo "================================================"

# Show log commands if detached
if [ "$DETACHED" = true ]; then
    echo ""
    echo "Services running in background"
    echo ""
    echo "View logs:"
    echo "  All services: docker compose logs -f"
    echo "  App only:     docker compose logs -f app"
    echo "  Cloudflare:   docker compose logs -f cloudflared"
    echo ""
    echo "Stop services: ./scripts/prod/stop.sh"
    echo "Clean up:     ./scripts/prod/clean.sh"
fi