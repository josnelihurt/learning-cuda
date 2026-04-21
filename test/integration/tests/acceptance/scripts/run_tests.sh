#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

curl -k -s https://localhost:8443/health > /dev/null 2>&1 || {
    echo "Service not running. Start with: ./scripts/dev/start.sh"
    exit 1
}

cd "$PROJECT_ROOT/integration/tests/acceptance"
go test -v -run TestFeatures -timeout 120s
