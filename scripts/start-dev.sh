#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

BUILD_FIRST=false

for arg in "$@"; do
    [[ "$arg" == "--build" ]] || [[ "$arg" == "-b" ]] && BUILD_FIRST=true
done

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
    bazel build //cpp_accelerator/ports/shared_lib:libcuda_processor_mock.so
    
    echo "Installing libraries..."
    mkdir -p .ignore/lib/cuda_learning
    cp bazel-bin/cpp_accelerator/ports/shared_lib/libcuda_processor.so .ignore/lib/cuda_learning/libcuda_processor_v$(cat cpp_accelerator/VERSION).so
    cp bazel-bin/cpp_accelerator/ports/shared_lib/libcuda_processor_mock.so .ignore/lib/cuda_learning/
    
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

    cat > .ignore/lib/cuda_learning/libcuda_processor_mock.so.json <<EOF
{
  "name": "Mock Image Processor",
  "version": "mock",
  "api_version": "2.0.0",
  "type": "mock",
  "build_date": "${DATE}",
  "build_commit": "${COMMIT}",
  "description": "Passthrough mock for fast development (no processing)"
}
EOF
    
    echo "Building backend with Go..."
    cd webserver && make build && cd ..
}

cd webserver/web
[ ! -d "node_modules" ] && npm install
cd "$PROJECT_ROOT"

cleanup() {
    echo "Stopping services..."
    kill $VITE_PID $GO_PID 2>/dev/null
    wait $VITE_PID $GO_PID 2>/dev/null
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

echo "Starting Go server..."
[ ! -f "bin/server" ] && {
    echo "Server binary not found, building..."
    cd webserver && make build && cd ..
}

./bin/server -webroot=webserver/web &
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
echo "Press Ctrl+C to stop"

wait $VITE_PID $GO_PID

