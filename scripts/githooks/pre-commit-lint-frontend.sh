#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Frontend Linter..."
cd "$PROJECT_ROOT/webserver/web"
if [ -d "node_modules" ]; then
    npm run lint || {
        echo "FAILED: Frontend linter"
        exit 1
    }
    echo "✓ Frontend linter passed"
else
    echo "ERROR: node_modules not found. Run: npm install"
    exit 1
fi

