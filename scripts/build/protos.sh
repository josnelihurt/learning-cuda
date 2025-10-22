#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

IMAGE_NAME="cuda-learning-bufgen:latest"
DOCKERFILE="bufgen.dockerfile"

echo "Building Protocol Buffers..."

if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Building bufgen Docker image..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" . || {
        echo "FAILED: Could not build bufgen image"
        exit 1
    }
    echo "Bufgen image built successfully"
fi

echo "Generating proto files..."
docker run --rm \
    -v "$PROJECT_ROOT:/workspace" \
    -u $(id -u):$(id -g) \
    "$IMAGE_NAME" generate || {
    echo "FAILED: Proto generation failed"
    exit 1
}

echo "Proto files generated successfully"
echo "  Go:         proto/gen/*.pb.go"
echo "  TypeScript: webserver/web/src/gen/*.ts"

