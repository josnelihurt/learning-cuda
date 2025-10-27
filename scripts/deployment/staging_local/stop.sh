#!/bin/bash
#
# Stop Production Services
#
# This script stops all production services including:
# - Application containers
# - Traefik reverse proxy
# - Cloudflare Tunnel (if running)
# - All supporting services
#
# Usage:
#   ./scripts/prod/stop.sh
#
# Note: Volumes are preserved (data not deleted)
# To remove volumes: ./scripts/prod/clean.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Stopping production services..."

# Stop all services including cloudflare profile
docker compose --profile cloudflare down

echo ""
echo "Production services stopped"
echo "Volumes preserved (data not deleted)"
echo ""
echo "To remove volumes: ./scripts/prod/clean.sh"