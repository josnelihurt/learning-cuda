#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--clean" ]; then
    echo "Clean option not implemented yet"
fi

echo "Starting deployment process..."
echo "Step 1/3: Initializing..."
if ! "${SCRIPT_DIR}/init.sh"; then
    echo "ERROR: Initialization failed"
    exit 1
fi

echo "Step 2/3: Synchronizing files..."
if ! "${SCRIPT_DIR}/sync.sh"; then
    echo "ERROR: File synchronization failed"
    exit 1
fi

echo "Step 3/3: Starting services..."
if ! "${SCRIPT_DIR}/start.sh"; then
    echo "ERROR: Service startup failed"
    exit 1
fi

echo "Deployment complete"
echo "Access: https://app-cuda-demo.josnelihurt.me"

