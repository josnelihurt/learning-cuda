#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

CLEAN_ALL=false
if [ "$1" = "--all" ]; then
    CLEAN_ALL=true
fi

echo "Cleaning build artifacts and test results..."

echo "  Cleaning Bazel cache..."
bazel clean --expunge

echo "  Cleaning Go cache..."
go clean -cache -modcache -testcache

echo "  Cleaning build outputs..."
rm -f webserver/cmd/server/server
rm -f bin/server

echo "  Cleaning generated proto files..."
rm -rf proto/gen/*

echo "  Cleaning test results..."
rm -rf webserver/web/.ignore/
rm -rf webserver/web/test-results/
rm -rf webserver/web/playwright-report/
sudo rm -rf integration/tests/acceptance/.ignore/ 2>/dev/null || rm -rf integration/tests/acceptance/.ignore/

echo "  Cleaning test data (video frames)..."
rm -rf data/test-data/video-frames/

echo "  Cleaning coverage reports..."
rm -rf coverage/frontend/
rm -rf coverage/golang/
rm -rf coverage/cpp/

if [ "$CLEAN_ALL" = true ]; then
    echo "  Cleaning SSL certificates (--all flag)..."
    rm -rf .secrets/localhost+2*.pem
fi

echo ""
if [ "$CLEAN_ALL" = true ]; then
    echo "Clean complete (including SSL certificates)"
    echo "Certificates will be regenerated on next ./scripts/dev/start.sh"
else
    echo "Clean complete (SSL certificates preserved)"
    echo "Use --all to also remove SSL certificates"
fi
