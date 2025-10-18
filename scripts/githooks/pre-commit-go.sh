#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Go Unit Tests..."
cd "$PROJECT_ROOT/webserver"
make test || {
    echo "FAILED: Go tests"
    exit 1
}
echo "✓ Go tests passed"

