#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

wait_for_grpc() {
    echo "Waiting for gRPC dev server on localhost:60061..."
    local timeout=30

    while [ "$timeout" -gt 0 ]; do
        if bash -c ">/dev/tcp/localhost/60061" 2>/dev/null; then
            echo "gRPC dev server is reachable on localhost:60061"
            return 0
        fi

        sleep 1
        timeout=$((timeout - 1))
    done

    echo "Error: gRPC dev server is not reachable on localhost:60061"
    echo "       Make sure the 'grpc-server-dev' service is healthy (docker-compose.dev.yml)"
    exit 1
}

ensure_writable() {
    local relative_path="$1"
    local absolute_path="${PROJECT_ROOT}/${relative_path}"
    local uid gid owner group
    uid="$(id -u)"
    gid="$(id -g)"

    if [ ! -d "$absolute_path" ]; then
        if ! mkdir -p "$absolute_path" 2>/dev/null; then
            if command -v docker >/dev/null 2>&1; then
                docker run --rm -v "${PROJECT_ROOT}:/workspace" alpine:3.19 \
                    sh -c "mkdir -p \"/workspace/${relative_path}\" && chown -R ${uid}:${gid} \"/workspace/${relative_path}\" && chmod -R u+rwX,g+rwX \"/workspace/${relative_path}\""
            else
                echo "Unable to create ${absolute_path}. Run 'sudo mkdir -p ${absolute_path}' and ensure it is owned by ${uid}:${gid}."
                exit 1
            fi
        fi
    fi

    if owner="$(stat -c '%u' "$absolute_path" 2>/dev/null)"; then
        group="$(stat -c '%g' "$absolute_path" 2>/dev/null)"
    else
        owner="$(stat -f '%u' "$absolute_path")"
        group="$(stat -f '%g' "$absolute_path")"
    fi
    if [ "$owner" != "$uid" ] || [ "$group" != "$gid" ] || [ ! -w "$absolute_path" ]; then
        if command -v docker >/dev/null 2>&1; then
            docker run --rm -v "${absolute_path}:/target" alpine:3.19 \
                sh -c "chown -R ${uid}:${gid} /target && chmod -R u+rwX,g+rwX /target"
        else
            echo "Unable to adjust permissions for ${absolute_path}. Run 'sudo chown -R ${uid}:${gid} ${absolute_path}'."
            exit 1
        fi
    fi
}

# Load development secrets
if [ -f ".secrets/development.env" ]; then
    echo "Loading development secrets from .secrets/development.env"
    source .secrets/development.env
else
    echo "WARNING: Development secrets file not found at .secrets/development.env"
    echo "Creating template file..."
    mkdir -p .secrets
    cp .secrets/development.env.example .secrets/development.env
    echo "Please edit .secrets/development.env with your actual development secrets"
fi

BUILD_FIRST=false
SHOW_HELP=false

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

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: ./scripts/dev/start.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build, -b    Build C++ libraries and Go backend before starting"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/dev/start.sh         # Start development server"
    echo "  ./scripts/dev/start.sh --build # Build + start development server"
    echo ""
    echo "Default processor library: 2.0.0 (CUDA)"
    exit 0
fi

if [ ! -f ".secrets/localhost+2.pem" ]; then
    echo "SSL certificates not found, generating with Docker..."
    ./scripts/docker/generate-certs.sh
    echo ""
fi

echo "Checking services (Jaeger + OTel Collector + Flipt)..."
if ! docker ps --format '{{.Names}}' | grep -q 'jaeger'; then
    echo "Starting services..."
    docker compose -f docker-compose.dev.yml up -d \
        jaeger \
        otel-collector \
        flipt \
        loki \
        promtail \
        grafana \
        mcp-grafana
    
    echo "Waiting for Jaeger to be healthy..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if docker ps --format '{{.Names}}\t{{.Status}}' | grep 'jaeger' | grep -q 'healthy'; then
            echo "Jaeger is ready!"
            break
        fi
        sleep 1
        timeout=$((timeout - 1))
    done
    if [ $timeout -eq 0 ]; then
        echo "Warning: Jaeger health check timeout. Continuing anyway..."
    fi
    
    echo "Waiting for Flipt to be ready..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8081/api/v1/health > /dev/null 2>&1; then
            echo "Flipt health check passed, waiting for full initialization..."
            sleep 3
            echo "Flipt is ready!"
            break
        fi
        sleep 1
        timeout=$((timeout - 1))
    done
    if [ $timeout -eq 0 ]; then
        echo "Warning: Flipt is not responding. Flag sync may fail..."
    fi
else
    echo "Services already running"
    
    # Verify Flipt is accessible
    if ! curl -s http://localhost:8081/api/v1/health > /dev/null 2>&1; then
        echo "Warning: Flipt is not responding at http://localhost:8081"
        echo "   Starting Flipt..."
        docker compose -f docker-compose.dev.yml up -d flipt
        sleep 5
    fi
fi

echo "Stopping previous application services..."
./scripts/dev/stop.sh 2>/dev/null || true

[ "$BUILD_FIRST" = true ] && {
    echo "Checking proto files..."
    if [ ! -f "proto/gen/image_processor_service.pb.go" ]; then
        ./scripts/build/protos.sh
    fi
    
    echo "Building C++ processor libraries..."
    bazel build //cpp_accelerator/ports/shared_lib:libcuda_processor.so
    
    echo "Installing libraries..."
    ensure_writable ".ignore/lib/cuda_learning"
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "dev")
    DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    VERSION=$(cat cpp_accelerator/VERSION)
    rm -f .ignore/lib/cuda_learning/libcuda_processor_v${VERSION}.so
    cp bazel-bin/cpp_accelerator/ports/shared_lib/libcuda_processor.so .ignore/lib/cuda_learning/libcuda_processor_v${VERSION}.so
    
    cat > .ignore/lib/cuda_learning/libcuda_processor_v${VERSION}.so.json <<EOF
{
  "name": "CUDA Image Processor",
  "version": "${VERSION}",
  "api_version": "2.1.0",
  "type": "gpu",
  "build_date": "${DATE}",
  "build_commit": "${COMMIT}",
  "description": "CUDA-accelerated image processing with CPU fallback"
    }
EOF
    
    echo "Building gRPC processor server..."
    bazel build //cpp_accelerator/ports/grpc:image_processor_grpc_server

    echo "Building backend with Go..."
    cd webserver && make build && cd ..
}

GRPC_SERVER_BIN="${PROJECT_ROOT}/bazel-bin/cpp_accelerator/ports/grpc/image_processor_grpc_server"

if [ ! -x "$GRPC_SERVER_BIN" ]; then
    echo "Error: gRPC server binary not found at ${GRPC_SERVER_BIN}"
    echo "       Run './scripts/dev/start.sh --build' to build it."
    exit 1
fi

echo "Starting local gRPC server..."
"$GRPC_SERVER_BIN" --listen_addr=0.0.0.0:60061 > /tmp/grpc-server.log 2>&1 &
GRPC_PID=$!

wait_for_grpc

cd webserver/web
[ ! -d "node_modules" ] && npm install
cd "$PROJECT_ROOT"

cleanup_on_error() {
    echo "Error detected, stopping services..."
    if [ -n "${GO_PID:-}" ]; then
        kill "$GO_PID" 2>/dev/null || true
    fi
    if [ -n "${VITE_PID:-}" ]; then
        kill "$VITE_PID" 2>/dev/null || true
    fi
    if [ -n "${GRPC_PID:-}" ]; then
        kill "$GRPC_PID" 2>/dev/null || true
    fi
    wait ${GO_PID:-} ${VITE_PID:-} ${GRPC_PID:-} 2>/dev/null || true
}

trap cleanup_on_error INT TERM

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

echo "Starting Go server..."
echo "Building Go server (always compile)..."
cd webserver && make build && cd ..

# Check if build was successful
if [ ! -f "./bin/server" ]; then
    echo "Build failed, exiting..."
    exit 1
fi

VERSION=$(cat cpp_accelerator/VERSION)
echo "Using CUDA processor library (version ${VERSION})"
./bin/server -config=config/config.dev.yaml > /tmp/goserver.log 2>&1 &
GO_PID=$!

sleep 2

! kill -0 $GO_PID 2>/dev/null && {
    echo "Error: Go server failed to start"
    kill $VITE_PID 2>/dev/null
    exit 1
}

echo ""
echo "Starting test report viewers..."
docker compose -f docker-compose.dev.yml --profile testing up -d e2e-report-viewer cucumber-report 2>/dev/null || true
docker compose -f docker-compose.dev.yml --profile coverage up -d coverage-report-viewer gopkgview 2>/dev/null || true
sleep 2

echo "================================================"
echo "Development server running:"
echo "  HTTPS:  https://localhost:8443"
echo "  Jaeger: http://localhost:16686"
echo "  Flipt:  http://localhost:8081"
echo ""
echo "Test Reports:"
echo "  BDD:      http://localhost:5050"
echo "  E2E:      http://localhost:5051"
echo "  Coverage: http://localhost:5052"
echo "  Go Deps:  http://localhost:5053"
echo "================================================"
echo "Dev mode - hot reload enabled"
echo "Observability & Feature Flags enabled"
echo "Processor: CUDA (version 2.0.0)"
echo ""
echo "Services running in background"
echo "  Vite PID: $VITE_PID"
echo "  Go Server PID: $GO_PID"
echo ""
echo "To stop services, run: ./scripts/dev/stop.sh"
echo "To view logs:"
echo "  Vite:      tail -f /tmp/vite.log"
echo "  Go Server: tail -f /tmp/goserver.log"

