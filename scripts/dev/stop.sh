#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Compose command configuration
COMPOSE_CMD="podman compose"
# COMPOSE_CMD="docker compose"

echo "Stopping services..."

# Stop app-dev container
$COMPOSE_CMD -f docker-compose.dev.yml stop app-dev 2>/dev/null && echo "Go server container stopped" || echo "Go server container not running"

pkill -f "vite" 2>/dev/null && echo "Vite stopped" || echo "Vite not running"

fuser -k 2019/tcp 2>/dev/null || true
fuser -k 8443/tcp 2>/dev/null || true

# Stop test report viewers
$COMPOSE_CMD -f docker-compose.dev.yml --profile testing stop e2e-report-viewer cucumber-report 2>/dev/null || true
$COMPOSE_CMD -f docker-compose.dev.yml --profile coverage stop coverage-report-viewer 2>/dev/null || true
echo "Test report viewers stopped"

sleep 1
echo "Services stopped"
