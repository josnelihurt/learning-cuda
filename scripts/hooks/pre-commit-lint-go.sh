#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Go Linter (Docker)..."
cd "$PROJECT_ROOT"

# Run the linter excluding CGO packages that require hardware
docker compose -f docker-compose.dev.yml --profile lint run --rm lint-golang || {
    echo "FAILED: Go linter"
    exit 1
}

echo "Go linter passed" // emoji-allowed

