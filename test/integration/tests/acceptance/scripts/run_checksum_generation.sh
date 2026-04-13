#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_URL="${1:-https://localhost:8443}"

echo "Generating checksums for test images..."
echo "Service URL: $SERVICE_URL"
echo ""

cd "$SCRIPT_DIR"
go run generate_checksums.go "$SERVICE_URL"

echo ""
echo "✓ Checksums generated successfully!"

