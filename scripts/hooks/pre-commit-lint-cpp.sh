#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "C++ Linter (Docker)..."
cd "$PROJECT_ROOT"
docker compose -f docker-compose.dev.yml --profile lint run --rm lint-cpp || {
    echo "FAILED: C++ linter"
    exit 1
}
echo "✓ C++ linter passed"

