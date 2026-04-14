#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

CLEAN_ALL=false
if [[ "${1:-}" == "--all" ]]; then
  CLEAN_ALL=true
fi

# Layout: application code under src/, integration tests under test/
GO_API_DIR="src/go_api"
FRONTEND_DIR="src/front-end"
INTEGRATION_ACCEPTANCE="test/integration/tests/acceptance"
COVERAGE_DIR="test/coverage"

echo "Cleaning build artifacts and test outputs..."

echo "  Bazel outputs and cache..."
bazel clean --expunge

echo "  Go build cache..."
go clean -cache -modcache -testcache

echo "  Compiled binaries (repo root + stray go build in cmd/server)..."
rm -f "${GO_API_DIR}/cmd/server/server"
rm -f bin/server

echo "  Generated protobuf (Go/C++); regenerate with ./scripts/build/protos.sh..."
rm -rf proto/gen/*

echo "  Frontend / Playwright / Vite scratch (local + Docker volume targets)..."
rm -rf .ignore/front-end/
rm -rf "${FRONTEND_DIR}/.ignore/"
rm -rf "${FRONTEND_DIR}/test-results/"
rm -rf "${FRONTEND_DIR}/playwright-report/"

echo "  BDD acceptance scratch (${INTEGRATION_ACCEPTANCE}/.ignore)..."
sudo rm -rf "${INTEGRATION_ACCEPTANCE}/.ignore/" 2>/dev/null || rm -rf "${INTEGRATION_ACCEPTANCE}/.ignore/"

echo "  Coverage report trees (${COVERAGE_DIR}/...)..."
rm -rf "${COVERAGE_DIR}/frontend/"
rm -rf "${COVERAGE_DIR}/golang/"
rm -rf "${COVERAGE_DIR}/cpp/"

if [[ -d coverage ]]; then
  echo "  Legacy root coverage/ (pre-restructure)..."
  rm -rf coverage/
fi

echo "  Test data (extracted video frames; re-extract with scripts/tools if needed)..."
rm -rf data/test-data/video-frames/

if [[ "$CLEAN_ALL" == true ]]; then
  echo "  Dev TLS material (--all)..."
  rm -f .secrets/localhost+2-key.pem .secrets/localhost+2.pem
fi

echo ""
if [[ "$CLEAN_ALL" == true ]]; then
  echo "Clean complete (including dev TLS certificates)."
  echo "Run ./scripts/docker/generate-certs.sh or ./scripts/dev/start.sh to recreate certs."
else
  echo "Clean complete (TLS certificates under .secrets/ preserved)."
  echo "Use --all to also remove .secrets/localhost+2*.pem"
fi
