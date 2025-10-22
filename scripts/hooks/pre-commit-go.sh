#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Go Unit Tests (with race detection)..."
cd "$PROJECT_ROOT"

# Use the unit tests script that handles CGO issues
"$PROJECT_ROOT/scripts/test/unit-tests.sh" || {
    echo "FAILED: Go tests (race conditions detected or tests failed)"
    exit 1
}

echo "OK: Go tests passed (no race conditions)"

