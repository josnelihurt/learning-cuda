#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

PROD_MODE=false
BUILD_FIRST=false

for arg in "$@"; do
    [[ "$arg" == "--prod" ]] || [[ "$arg" == "-p" ]] && PROD_MODE=true
    [[ "$arg" == "--build" ]] || [[ "$arg" == "-b" ]] && BUILD_FIRST=true
done

[ ! -f ".secrets/localhost+2.pem" ] && {
    echo "Error: SSL certificates not found. Run: ./scripts/setup-ssl.sh"
    exit 1
}

! command -v caddy &> /dev/null && {
    echo "Error: Caddy not installed"
    exit 1
}

echo "Stopping previous services..."
./scripts/kill-services.sh 2>/dev/null || true

[ "$BUILD_FIRST" = true ] && {
    echo "Checking proto files..."
    if [ ! -f "proto/gen/image_processing.pb.go" ]; then
        echo "Generating proto files..."
        docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
    fi
    
    echo "Building backend..."
    bazel build //webserver/cmd/server:server //cpp_accelerator/ports/cgo:cgo_api
}

if [ "$PROD_MODE" = false ]; then
    cd webserver/web
    [ ! -d "node_modules" ] && npm install
    cd "$PROJECT_ROOT"
    
    cleanup() {
        echo "Stopping services..."
        kill $VITE_PID $GO_PID $CADDY_PID 2>/dev/null
        wait $VITE_PID $GO_PID $CADDY_PID 2>/dev/null
    }
    
    trap cleanup EXIT INT TERM
    
    echo "Starting Vite (hot reload)..."
    cd webserver/web
    npm run dev > /tmp/vite.log 2>&1 &
    VITE_PID=$!
    cd "$PROJECT_ROOT"
    sleep 2
    
    ! kill -0 $VITE_PID 2>/dev/null && {
        echo "Error: Vite failed to start"
        cat /tmp/vite.log
        exit 1
    }
    
    WEBROOT_PATH="$PROJECT_ROOT/webserver/web"
    bazel-bin/webserver/cmd/server/server_/server -dev -webroot="$WEBROOT_PATH" &
    GO_PID=$!
else
    [ "$BUILD_FIRST" = true ] && ./scripts/build-frontend.sh
    
    cleanup() {
        echo "Stopping services..."
        kill $GO_PID $CADDY_PID 2>/dev/null
        wait $GO_PID $CADDY_PID 2>/dev/null
    }
    
    trap cleanup EXIT INT TERM
    
    WEBROOT_PATH="$PROJECT_ROOT/webserver/web"
    bazel-bin/webserver/cmd/server/server_/server -webroot="$WEBROOT_PATH" &
    GO_PID=$!
fi

sleep 2

! kill -0 $GO_PID 2>/dev/null && {
    echo "Error: Go server failed to start"
    exit 1
}

caddy run --config Caddyfile --adapter caddyfile 2>&1 | sed 's/^/[Caddy] /' &
CADDY_PID=$!
sleep 2

! kill -0 $CADDY_PID 2>/dev/null && {
    echo "Error: Caddy failed to start"
    [ "$PROD_MODE" = false ] && kill $VITE_PID 2>/dev/null
    kill $GO_PID 2>/dev/null
    exit 1
}

echo ""
echo "https://localhost:8443"
[ "$PROD_MODE" = false ] && echo "Dev mode - hot reload enabled" || echo "Production mode"
echo "Press Ctrl+C to stop"

[ "$PROD_MODE" = false ] && wait $VITE_PID $GO_PID $CADDY_PID || wait $GO_PID $CADDY_PID
