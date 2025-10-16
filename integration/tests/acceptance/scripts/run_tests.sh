#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "═══════════════════════════════════════════════════════════"
echo "  CUDA Learning - Integration Tests"
echo "═══════════════════════════════════════════════════════════"
echo ""

SERVICE_READY=false

check_service() {
    if curl -k -s https://localhost:8443/health > /dev/null 2>&1; then
        SERVICE_READY=true
        echo "✓ Service is running at https://localhost:8443"
    else
        SERVICE_READY=false
        echo "✗ Service is not running"
    fi
}

check_service

if [ "$SERVICE_READY" = false ]; then
    echo ""
    echo "Please start the service first:"
    echo "  ./scripts/start-dev.sh"
    echo ""
    exit 1
fi

CHECKSUMS_FILE="$PROJECT_ROOT/integration/tests/acceptance/testdata/checksums.json"

if [ ! -f "$CHECKSUMS_FILE" ] || [ "$(cat "$CHECKSUMS_FILE" | grep -c "pending")" -gt 0 ]; then
    echo ""
    echo "⚠ Checksums not generated yet"
    echo "Generating checksums..."
    echo ""
    cd "$SCRIPT_DIR"
    ./run_checksum_generation.sh
    echo ""
fi

echo "─────────────────────────────────────────────────────────────"
echo "  Running BDD Tests"
echo "─────────────────────────────────────────────────────────────"
echo ""

cd "$PROJECT_ROOT/integration/tests/acceptance"

go test -v -run TestFeatures -timeout 120s

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Tests Complete!"
echo "═══════════════════════════════════════════════════════════"

