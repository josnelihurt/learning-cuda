#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "${1:-}" = "--clean" ]; then
    echo "Clean option not implemented yet"
fi

"${SCRIPT_DIR}/init.sh"
"${SCRIPT_DIR}/sync.sh"
"${SCRIPT_DIR}/start.sh"

echo "Deployment complete"
echo "Access: https://app-cuda-demo.josnelihurt.me"

