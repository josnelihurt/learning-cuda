#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEB_DIR="$PROJECT_ROOT/webserver/web"

echo "Building frontend..."

cd "$WEB_DIR"

[ ! -d "node_modules" ] && npm install

npm run build

echo "Build complete. Output: webserver/web/static/js/dist/"
ls -lh static/js/dist/app.*.js 2>/dev/null || echo "Warning: bundle not found"
