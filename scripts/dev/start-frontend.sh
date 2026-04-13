#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: ./scripts/dev/start-frontend.sh"
    echo ""
    echo "Local dev (split ports):"
    echo "  - UI: https://localhost:3000/react and https://localhost:3000/lit (Vite pretty paths)"
    echo "  - API/WebSocket/Connect: Go TLS https://localhost:8443 (set VITE_API_ORIGIN if needed)"
    echo ""
    echo "Production: Traefik routes to Nginx (web-frontend) for static HTML; go-app serves /api, /ws, etc."
    echo "Nginx serves /react and /lit; Go does not serve frontend HTML."
    echo ""
    echo "Start the backend first: ./scripts/dev/start.sh"
    echo "Optional: VITE_API_ORIGIN (default https://localhost:8443)"
    exit 0
fi

if [ ! -f ".secrets/localhost+2.pem" ]; then
    echo "SSL certificates not found. Run ./scripts/dev/start.sh once (or ./scripts/docker/generate-certs.sh)."
    exit 1
fi

cd "$PROJECT_ROOT/front-end"
[ ! -d "node_modules" ] && npm install

echo "Dev: UI https://localhost:3000/lit and https://localhost:3000/react | API https://localhost:8443 (VITE_API_ORIGIN) | Prod UI: Nginx, not Go"
echo "Starting Vite at https://localhost:3000 (API proxy -> \${VITE_API_ORIGIN:-https://localhost:8443})"
exec npm run dev
