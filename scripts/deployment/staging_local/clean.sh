#!/bin/bash
#
# Clean Staging Environment
#
# This script removes all staging containers, volumes, and optionally images.
# Use with caution - this action cannot be undone!
#
# Usage:
#   ./scripts/deployment/staging_local/clean.sh           # Remove containers and volumes
#   ./scripts/deployment/staging_local/clean.sh --images  # Also remove Docker images
#
# WARNING: This will delete all data in volumes!
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

REMOVE_IMAGES=false
for arg in "$@"; do
    if [[ "$arg" == "--images" ]]; then
        REMOVE_IMAGES=true
    fi
done

echo "=========================================="
echo "WARNING: Staging Environment Cleanup"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - All staging containers"
echo "  - All staging volumes (data will be lost!)"
echo "  - All staging networks"
if [ "$REMOVE_IMAGES" = true ]; then
    echo "  - All staging Docker images"
fi
echo ""
echo "This action cannot be undone!"
echo ""
read -p "Are you sure you want to continue? (yes/no): " -r
echo

if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo "Cleaning staging environment..."

if [ "$REMOVE_IMAGES" = true ]; then
    docker compose -f docker-compose.staging.yml down --volumes --rmi all --remove-orphans
    echo ""
    echo "Removed: containers, volumes, images, networks"
else
    docker compose -f docker-compose.staging.yml down --volumes --remove-orphans
    echo ""
    echo "Removed: containers, volumes, networks"
    echo "Images preserved (use --images to remove)"
fi

echo ""
echo "Staging environment cleaned"
echo ""
echo "To start fresh: ./scripts/deployment/staging_local/start.sh"
