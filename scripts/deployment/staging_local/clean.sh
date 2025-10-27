#!/bin/bash
#
# Clean Production Environment
#
# This script removes all production containers, volumes, and optionally images.
# Use with caution - this action cannot be undone!
#
# Usage:
#   ./scripts/prod/clean.sh           # Remove containers and volumes
#   ./scripts/prod/clean.sh --images  # Also remove Docker images
#
# WARNING: This will delete all data in volumes!
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

REMOVE_IMAGES=false
for arg in "$@"; do
    if [[ "$arg" == "--images" ]]; then
        REMOVE_IMAGES=true
    fi
done

echo "=========================================="
echo "WARNING: Production Environment Cleanup"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - All production containers"
echo "  - All production volumes (data will be lost!)"
echo "  - All production networks"
if [ "$REMOVE_IMAGES" = true ]; then
    echo "  - All production Docker images"
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
echo "Cleaning production environment..."

if [ "$REMOVE_IMAGES" = true ]; then
    docker compose --profile cloudflare down --volumes --rmi all --remove-orphans
    echo ""
    echo "Removed: containers, volumes, images, networks"
else
    docker compose --profile cloudflare down --volumes --remove-orphans
    echo ""
    echo "Removed: containers, volumes, networks"
    echo "Images preserved (use --images to remove)"
fi

echo ""
echo "Production environment cleaned"
echo ""
echo "To start fresh: ./scripts/prod/start.sh"