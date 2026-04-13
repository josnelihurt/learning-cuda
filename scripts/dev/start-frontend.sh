#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: ./scripts/dev/start-frontend.sh"
    echo ""
    echo "Runs the Vite dev server (HTTPS :3000). Proxies API/WebSocket traffic to the Go backend."
    echo "Start the backend first: ./scripts/dev/start.sh"
    echo ""
    echo "Optional: VITE_API_ORIGIN (default https://localhost:8443)"
    exit 0
fi

if [ ! -f ".secrets/localhost+2.pem" ]; then
    echo "SSL certificates not found. Run ./scripts/dev/start.sh once (or ./scripts/docker/generate-certs.sh)."
    exit 1
fi

cd "$PROJECT_ROOT/front-end"
[ ! -d "node_modules" ] && npm install

echo "Starting Vite at https://localhost:3000 (API proxy -> \${VITE_API_ORIGIN:-https://localhost:8443})"
exec npm run dev
