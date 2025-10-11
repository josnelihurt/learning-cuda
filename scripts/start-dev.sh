#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Parse arguments
BUILD_FIRST=false
if [[ "$1" == "--build" ]] || [[ "$1" == "-b" ]]; then
    BUILD_FIRST=true
fi

if [ ! -f ".secrets/localhost+2.pem" ] || [ ! -f ".secrets/localhost+2-key.pem" ]; then
    echo "Error: SSL certificates not found"
    echo "Run: ./scripts/setup-ssl.sh"
    exit 1
fi

if ! command -v caddy &> /dev/null; then
    echo "Error: Caddy not installed"
    echo "Install from: https://caddyserver.com/download"
    exit 1
fi

# Check if another Caddy instance is running
if pgrep -x "caddy" > /dev/null; then
    echo "Warning: Caddy already running"
    read -p "Stop it with 'pkill caddy' or continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting CUDA Image Processor"
echo ""

if [ "$BUILD_FIRST" = true ]; then
    echo "Building project..."
    bazel build //webserver/cmd/server:server //cpp_accelerator/ports/cgo:cgo_api
    echo "Build complete"
    echo ""
fi

echo "Generating version..."
./scripts/generate-version.sh
echo ""

cleanup() {
    echo ""
    echo "Stopping services..."
    kill $GO_PID $CADDY_PID 2>/dev/null
    wait $GO_PID $CADDY_PID 2>/dev/null
}

trap cleanup EXIT INT TERM

echo "Starting Go server (dev mode)..."
WEBROOT_PATH="$PROJECT_ROOT/webserver/web"
bazel-bin/webserver/cmd/server/server_/server -dev -webroot="$WEBROOT_PATH" &
GO_PID=$!
sleep 2

if ! kill -0 $GO_PID 2>/dev/null; then
    echo "Error: Go server failed to start"
    exit 1
fi

echo "Starting Caddy..."
caddy run --config Caddyfile --adapter caddyfile 2>&1 | sed 's/^/[Caddy] /' &
CADDY_PID=$!
sleep 2

if ! kill -0 $CADDY_PID 2>/dev/null; then
    echo "Error: Caddy failed to start"
    kill $GO_PID 2>/dev/null
    exit 1
fi

echo ""
echo "Services running:"
echo "  HTTPS: https://localhost:8443"
echo "  HTTP:  http://localhost:8000 (redirects)"
echo "  Direct: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop"
echo ""

wait $GO_PID $CADDY_PID
