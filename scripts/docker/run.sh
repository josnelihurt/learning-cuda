#!/bin/bash
#
# Production Docker Deployment
#
# This script runs the full production stack with:
# - CUDA-accelerated backend in Docker
# - Traefik reverse proxy (HTTPS with auto redirect)
# - NVIDIA GPU passthrough
# - Multi-stage optimized build
#
# Usage:
#   ./scripts/docker/run.sh           # Run in foreground (with logs)
#   ./scripts/docker/run.sh -d        # Run detached (background)
#   ./scripts/docker/run.sh --detach  # Run detached (background)
#
# Access:
#   https://localhost        - Main application (HTTP auto-redirects)
#   http://localhost:8081    - Traefik dashboard
#
# Stop:
#   docker compose down
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

./scripts/docker/validate-env.sh

if [ $? -ne 0 ]; then
    exit 1
fi

DETACHED=false
for arg in "$@"; do
    if [[ "$arg" == "-d" ]] || [[ "$arg" == "--detach" ]]; then
        DETACHED=true
    fi
done

if [ "$DETACHED" = true ]; then
    docker compose up -d --build
    echo ""
    echo "Containers started. Access at https://localhost"
    echo "Traefik dashboard: http://localhost:8081"
    echo ""
    echo "View logs: docker compose logs -f app"
    echo "Stop: docker compose down"
else
    docker compose up --build
fi
