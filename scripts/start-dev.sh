#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

BUILD_FIRST=false
USE_MOCK=false
SHOW_HELP=false

for arg in "$@"; do
    case "$arg" in
        --build|-b)
            BUILD_FIRST=true
            ;;
        --mock|-m)
            USE_MOCK=true
            ;;
        --help|-h)
            SHOW_HELP=true
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: ./scripts/start-dev.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build, -b    Build C++ libraries and Go backend before starting"
    echo "  --mock, -m     Use mock processor library (default: real CUDA library)"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/start-dev.sh                # Start with real CUDA library"
    echo "  ./scripts/start-dev.sh --build        # Build + start with real CUDA"
    echo "  ./scripts/start-dev.sh --mock         # Start with mock (no CUDA)"
    echo "  ./scripts/start-dev.sh --build --mock # Build + start with mock"
    echo ""
    echo "Default processor library: 2.0.0 (real CUDA)"
    echo "With --mock: mock (passthrough, fast development)"
    exit 0
fi

[ ! -f ".secrets/localhost+2.pem" ] && {
    echo "Error: SSL certificates not found."
    echo "Generate certificates with:"
    echo "  mkdir -p .secrets && cd .secrets"
    echo "  mkcert localhost 127.0.0.1 ::1"
    exit 1
}

echo "Checking services (Jaeger + OTel Collector + Flipt)..."
if ! docker ps --format '{{.Names}}' | grep -q 'jaeger-dev'; then
    echo "Starting services..."
    docker compose -f docker-compose.dev.yml up -d
    
    echo "Waiting for Jaeger to be healthy..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if docker ps --format '{{.Names}}\t{{.Status}}' | grep 'jaeger-dev' | grep -q 'healthy'; then
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
./scripts/kill-services.sh 2>/dev/null || true

[ "$BUILD_FIRST" = true ] && {
    echo "Checking proto files..."
    if [ ! -f "proto/gen/image_processor_service.pb.go" ]; then
        echo "Generating proto files..."
        docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
    fi
    
    echo "Building C++ processor libraries..."
    bazel build //cpp_accelerator/ports/shared_lib:libcuda_processor.so
    
    echo "Installing libraries..."
    mkdir -p .ignore/lib/cuda_learning
    cp bazel-bin/cpp_accelerator/ports/shared_lib/libcuda_processor.so .ignore/lib/cuda_learning/libcuda_processor_v$(cat cpp_accelerator/VERSION).so
    
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "dev")
    DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    VERSION=$(cat cpp_accelerator/VERSION)
    
    cat > .ignore/lib/cuda_learning/libcuda_processor_v${VERSION}.so.json <<EOF
{
  "name": "CUDA Image Processor",
  "version": "${VERSION}",
  "api_version": "2.0.0",
  "type": "gpu",
  "build_date": "${DATE}",
  "build_commit": "${COMMIT}",
  "description": "CUDA-accelerated image processing with CPU fallback"
}
EOF
    
    echo "Building backend with Go..."
    cd webserver && make build && cd ..
}

cd webserver/web
[ ! -d "node_modules" ] && npm install
cd "$PROJECT_ROOT"

cleanup_on_error() {
    echo "Error detected, stopping services..."
    kill $VITE_PID $GO_PID 2>/dev/null
    wait $VITE_PID $GO_PID 2>/dev/null
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
[ ! -f "bin/server" ] && {
    echo "Server binary not found, building..."
    cd webserver && make build && cd ..
}

if [ "$USE_MOCK" = true ]; then
    echo "Using MOCK processor library (fast, no CUDA)"
    export CUDA_PROCESSOR_PROCESSOR_DEFAULT_LIBRARY=mock
    ./bin/server -webroot=webserver/web > /tmp/goserver.log 2>&1 &
else
    echo "Using REAL CUDA processor library (version 2.0.0)"
    ./bin/server -webroot=webserver/web > /tmp/goserver.log 2>&1 &
fi
GO_PID=$!

sleep 2

! kill -0 $GO_PID 2>/dev/null && {
    echo "Error: Go server failed to start"
    kill $VITE_PID 2>/dev/null
    exit 1
}

echo ""
echo "Development server running:"
echo "  HTTP:   http://localhost:8080"
echo "  HTTPS:  https://localhost:8443"
echo "  Jaeger: http://localhost:16686"
echo "  Flipt:  http://localhost:8081"
echo ""
echo "Dev mode - hot reload enabled"
echo "Observability & Feature Flags enabled"
if [ "$USE_MOCK" = true ]; then
    echo "Processor: MOCK (passthrough, no CUDA)"
else
    echo "Processor: REAL CUDA (version 2.0.0)"
fi
echo ""
echo "Services running in background"
echo "  Vite PID: $VITE_PID"
echo "  Go Server PID: $GO_PID"
echo ""
echo "To stop services, run: ./scripts/kill-services.sh"
echo "To view logs:"
echo "  Vite:      tail -f /tmp/vite.log"
echo "  Go Server: docker logs -f \$(docker ps -q --filter ancestor=cuda-learning)"

