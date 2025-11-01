#!/bin/bash
#
# Stop Staging Services
#
# This script stops all staging services including:
# - Application containers
# - Traefik reverse proxy
# - All supporting services
#
# Usage:
#   ./scripts/deployment/staging_local/stop.sh
#
# Note: Volumes are preserved (data not deleted)
# To remove volumes: ./scripts/deployment/staging_local/clean.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Stopping staging services..."

docker compose -f docker-compose.staging.yml down

echo ""
echo "Staging services stopped"
echo "Volumes preserved (data not deleted)"
echo ""
echo "To remove volumes: ./scripts/deployment/staging_local/clean.sh"
