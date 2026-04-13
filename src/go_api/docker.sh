#!/bin/bash

set -e

case "$1" in
  build)
    echo "Building Docker image..."
    bazel build //src/go_api:server_image
    echo "Build complete"
    ;;
    
  load)
    echo "Loading image into Docker..."
    bazel run //src/go_api:server_load
    echo "Image loaded as cuda-webserver:latest"
    ;;
    
  run)
    echo "Running container on port 8080..."
    docker run -p 8080:8080 cuda-webserver:latest
    ;;
    
  run-bg)
    echo "Starting container in background..."
    CONTAINER_ID=$(docker run -d -p 8080:8080 cuda-webserver:latest)
    echo "Container started: $CONTAINER_ID"
    echo "Access at http://localhost:8080"
    echo "Stop with: docker stop $CONTAINER_ID"
    ;;
    
  all)
    echo "Building and loading image..."
    bazel build //src/go_api:server_image
    bazel run //src/go_api:server_load
    echo "Ready. Run with: ./src/go_api/docker.sh run"
    ;;
    
  *)
    echo "Usage: $0 {build|load|run|run-bg|all}"
    echo ""
    echo "Commands:"
    echo "  build   - Build Docker image with Bazel"
    echo "  load    - Load image into Docker daemon"
    echo "  run     - Run container (foreground)"
    echo "  run-bg  - Run container (background)"
    echo "  all     - Build and load in one step"
    exit 1
    ;;
esac
