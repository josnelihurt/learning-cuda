#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CUDA_LEARNING_RUNTIME_DIR="${CUDA_LEARNING_RUNTIME_DIR:-/tmp/cuda-learning}"
DEV_PID_DIR="${CUDA_LEARNING_RUNTIME_DIR}/PIDs"
DEV_PID_GRPC="${DEV_PID_DIR}/grpc.pid"
DEV_PID_GO="${DEV_PID_DIR}/go.pid"
DEV_PID_VITE="${DEV_PID_DIR}/vite.pid"

cd "$PROJECT_ROOT"

echo "Stopping services..."

kill_from_pid_file() {
    local pid_file=$1
    local label=$2
    local pid
    if [ ! -f "$pid_file" ]; then
        return 0
    fi
    IFS= read -r pid <"$pid_file" || pid=
    if [ -n "$pid" ] && kill "$pid" 2>/dev/null; then
        echo "$label stopped (pid $pid)"
    fi
    rm -f "$pid_file"
}

kill_from_pid_file "$DEV_PID_GRPC" " accelerator client"
kill_from_pid_file "$DEV_PID_GO" " Go server"
kill_from_pid_file "$DEV_PID_VITE" " Vite"

pkill -f "accelerator_control_client" 2>/dev/null && echo "accelerator client stopped (fallback)" || true

# Go server may be named bin/server; PID file can be stale after crash.
pkill -f "${PROJECT_ROOT}/bin/server" 2>/dev/null && echo "Go server stopped (fallback)" || true

pkill -f "vite" 2>/dev/null && echo "Vite stopped (fallback)" || echo "Vite not running"

fuser -k 2019/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
fuser -k 8443/tcp 2>/dev/null || true
fuser -k 60062/tcp 2>/dev/null || true

sleep 1
echo "Services stopped"
