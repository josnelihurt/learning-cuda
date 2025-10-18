#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Frontend Unit Tests..."
cd "$PROJECT_ROOT/webserver/web"
npm test -- --run || {
    echo "FAILED: Frontend tests"
    exit 1
}
echo "✓ Frontend tests passed"

