#!/bin/bash
# Helper script for Docker operations

set -e

case "$1" in
  build)
    echo "🔨 Building Docker image..."
    bazel build //webserver:server_image
    echo "✅ Image built successfully!"
    ;;
    
  load)
    echo "📦 Loading image into Docker..."
    bazel run //webserver:server_load
    echo "✅ Image loaded as cuda-webserver:latest"
    ;;
    
  run)
    echo "🚀 Running container on port 8080..."
    docker run -p 8080:8080 cuda-webserver:latest
    ;;
    
  run-bg)
    echo "🚀 Running container in background..."
    CONTAINER_ID=$(docker run -d -p 8080:8080 cuda-webserver:latest)
    echo "✅ Container started: $CONTAINER_ID"
    echo "📷 Open http://localhost:8080 in your browser"
    echo "🛑 Stop with: docker stop $CONTAINER_ID"
    ;;
    
  all)
    echo "🔄 Build and load image..."
    bazel build //webserver:server_image
    bazel run //webserver:server_load
    echo "✅ Ready! Run with: ./webserver/docker.sh run"
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

