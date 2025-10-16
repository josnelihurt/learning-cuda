#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

curl -k -s https://localhost:8443/health > /dev/null 2>&1 || {
    echo "Service not running. Start with: ./scripts/start-dev.sh"
    exit 1
}

CHECKSUMS_FILE="$PROJECT_ROOT/integration/tests/acceptance/testdata/checksums.json"

if [ ! -f "$CHECKSUMS_FILE" ] || grep -q "pending" "$CHECKSUMS_FILE" 2>/dev/null; then
    cd "$SCRIPT_DIR"
    ./run_checksum_generation.sh
fi

cd "$PROJECT_ROOT/integration/tests/acceptance"
go test -v -run TestFeatures -timeout 120s
