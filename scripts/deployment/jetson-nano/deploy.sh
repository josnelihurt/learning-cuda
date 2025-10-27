#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--clean" ]; then
    "${SCRIPT_DIR}/clean.sh"
fi

"${SCRIPT_DIR}/init.sh"
"${SCRIPT_DIR}/sync.sh"
"${SCRIPT_DIR}/start.sh"

echo "Deployment complete"
echo "Access: https://app-cuda-demo.josnelihurt.me"
