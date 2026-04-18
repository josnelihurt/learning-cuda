#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CUDA_LEARNING_RUNTIME_DIR="${CUDA_LEARNING_RUNTIME_DIR:-/tmp/cuda-learning}"
DEV_PID_DIR="${CUDA_LEARNING_RUNTIME_DIR}/PIDs"
DEV_LOG_DIR="${CUDA_LEARNING_RUNTIME_DIR}/logs"

DEV_PID_GRPC="${DEV_PID_DIR}/grpc.pid"
DEV_PID_GO="${DEV_PID_DIR}/go.pid"
DEV_PID_VITE="${DEV_PID_DIR}/vite.pid"

DEV_LOG_GRPC="${DEV_LOG_DIR}/grpc-server.log"
DEV_LOG_GO="${DEV_LOG_DIR}/goserver.log"
DEV_LOG_VITE="${DEV_LOG_DIR}/vite.log"

cd "$PROJECT_ROOT"

BUILD_FIRST=false
SHOW_HELP=false
GRPC_PID=
GO_PID=
VITE_PID=

die() {
    echo "Error: $*" >&2
    exit 1
}

parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --build|-b)
                BUILD_FIRST=true
                ;;
            --help|-h)
                SHOW_HELP=true
                ;;
        esac
    done
}

print_help() {
    echo "Usage: ./scripts/dev/start.sh [OPTIONS]"
    echo ""
    echo "Starts the C++ gRPC processor, Go API (HTTPS), and Vite dev server."
    echo ""
    echo "Local dev (split ports):"
    echo "  - UI:  https://localhost:3000/react and https://localhost:3000/lit (Vite pretty paths)"
    echo "  - API: https://localhost:8443 — REST, WebSocket, Connect (Vite proxies to Go via VITE_API_ORIGIN)"
    echo ""
    echo "Production: Traefik routes to Nginx (web-frontend) for static HTML; the Go app serves /api, WebRTC signaling, etc."
    echo "Nginx serves /react and /lit; Go does not serve frontend HTML in production."
    echo ""
    echo "VITE_API_ORIGIN defaults to https://localhost:8443 when unset."
    echo "Vite only (Go already running): cd src/front-end && npm run dev (needs .secrets/localhost+2*.pem)."
    echo ""
    echo "Options:"
    echo "  --build, -b    Build C++ gRPC server and Go backend before starting"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/dev/start.sh         # Full stack"
    echo "  ./scripts/dev/start.sh --build # Build + full stack"
    exit 0
}

require_secrets() {
    [ -f ".secrets/development.env" ] || {
        echo "Error: .secrets/development.env not found." >&2
        echo "      Copy .secrets/development.env.example to .secrets/development.env and fill in values." >&2
        exit 1
    }
    echo "Loading development secrets from .secrets/development.env"
    source .secrets/development.env
}

ensure_tls_certs() {
    if [ ! -f ".secrets/localhost+2.pem" ]; then
        echo "SSL certificates not found, generating..."
        ./scripts/docker/generate-certs.sh
        echo ""
    fi
}

ensure_state_dirs() {
    mkdir -p "$DEV_PID_DIR" "$DEV_LOG_DIR"
}

run_optional_build() {
    if [ "$BUILD_FIRST" != true ]; then
        return 0
    fi
    echo "Checking proto files..."
    if [ ! -f "proto/gen/image_processor_service.pb.go" ]; then
        ./scripts/build/protos.sh
    fi

    echo "Building gRPC processor server..."
    bazel build //src/cpp_accelerator/ports/grpc:image_processor_grpc_server

    echo "Building backend with Go..."
    (cd src/go_api && make build)
}

require_grpc_binary() {
    GRPC_SERVER_BIN="${PROJECT_ROOT}/bazel-bin/src/cpp_accelerator/ports/grpc/image_processor_grpc_server"
    if [ ! -x "$GRPC_SERVER_BIN" ]; then
        echo "Error: gRPC server binary not found at ${GRPC_SERVER_BIN}" >&2
        echo "       Run './scripts/dev/start.sh --build' to build it." >&2
        exit 1
    fi
}

start_grpc() {
    GRPC_SERVER_BIN="${PROJECT_ROOT}/bazel-bin/src/cpp_accelerator/ports/grpc/image_processor_grpc_server"
    echo "Starting gRPC (C++) server..."
    "$GRPC_SERVER_BIN" --listen_addr=0.0.0.0:60061 >"$DEV_LOG_GRPC" 2>&1 &
    GRPC_PID=$!
    echo "$GRPC_PID" >"$DEV_PID_GRPC"
}

start_go() {
    echo "Starting Go server..."
    echo "Building Go server..."
    (cd "$PROJECT_ROOT/src/go_api" && make build) || die "Go build failed"

    [ -f "$PROJECT_ROOT/bin/server" ] || die "Go build did not produce bin/server"

    "$PROJECT_ROOT/bin/server" -config=config/config.dev.yaml >"$DEV_LOG_GO" 2>&1 &
    GO_PID=$!
    echo "$GO_PID" >"$DEV_PID_GO"

    kill -0 "$GO_PID" 2>/dev/null || die "Go server failed to start"
}

start_vite() {
    cd "$PROJECT_ROOT/src/front-end"
    [ ! -d "node_modules" ] && npm install

    echo "Dev: UI  https://localhost:3000 | API https://localhost:8443 (VITE_API_ORIGIN) | Prod UI: Nginx, not Go"
    echo "Starting Vite at https://localhost:3000 (API proxy -> ${VITE_API_ORIGIN:-https://localhost:8443})"
    npm run dev >"$DEV_LOG_VITE" 2>&1 &
    VITE_PID=$!
    echo "$VITE_PID" >"$DEV_PID_VITE"
    cd "$PROJECT_ROOT"
}

cleanup_on_signal() {
    echo "Stopping services..."
    [ -n "${VITE_PID:-}" ] && kill "$VITE_PID" 2>/dev/null || true
    [ -n "${GO_PID:-}" ] && kill "$GO_PID" 2>/dev/null || true
    [ -n "${GRPC_PID:-}" ] && kill "$GRPC_PID" 2>/dev/null || true
    wait ${VITE_PID:-} ${GO_PID:-} ${GRPC_PID:-} 2>/dev/null || true
    rm -f "$DEV_PID_GRPC" "$DEV_PID_GO" "$DEV_PID_VITE"
}

print_summary() {
    echo "================================================"
    echo "Dev stack:"
    echo "  UI (Vite):   https://localhost:3000"
    echo "  API (HTTPS): https://localhost:8443"
    echo "  gRPC (C++):  localhost:60061"
    echo "================================================"
    echo ""
    echo "  gRPC server PID: $GRPC_PID ($DEV_PID_GRPC)"
    echo "  Go server PID:   $GO_PID ($DEV_PID_GO)"
    echo "  Vite PID:        $VITE_PID ($DEV_PID_VITE)"
    echo ""
    echo "To stop: ./scripts/dev/stop.sh"
    echo "Logs:"
    echo "  tail -f $DEV_LOG_GRPC"
    echo "  tail -f $DEV_LOG_GO"
    echo "  tail -f $DEV_LOG_VITE"
}

main() {
    parse_args "$@"
    [ "$SHOW_HELP" = true ] && print_help

    require_secrets
    ensure_tls_certs

    echo "Stopping previous dev processes..."
    ./scripts/dev/stop.sh 2>/dev/null || true

    run_optional_build
    require_grpc_binary
    ensure_state_dirs

    start_grpc
    trap cleanup_on_signal INT TERM

    start_go
    start_vite

    print_summary
}

main "$@"
