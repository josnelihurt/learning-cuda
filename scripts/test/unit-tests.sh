#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse command line arguments
SKIP_GOLANG=false
SKIP_FRONTEND=false
SKIP_CPP=false

for arg in "$@"; do
  case $arg in
    --skip-golang)
      SKIP_GOLANG=true
      shift
      ;;
    --skip-frontend)
      SKIP_FRONTEND=true
      shift
      ;;
    --skip-cpp)
      SKIP_CPP=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --skip-golang      Skip Golang unit tests"
      echo "  --skip-frontend    Skip Frontend unit tests"
      echo "  --skip-cpp         Skip C++ unit tests"
      echo ""
      echo "Examples:"
      echo "  $0                        # Run all tests"
      echo "  $0 --skip-golang          # Skip Golang only"
      echo "  $0 --skip-frontend --skip-cpp  # Run Golang only"
      exit 0
      ;;
  esac
done

echo "=================================="
echo "Running Unit Tests"
echo "=================================="
echo ""

# Golang Unit Tests
if [ "$SKIP_GOLANG" = false ]; then
  echo "[1/3] Running Go Unit Tests (with race detection)..."
  echo "=================================================="
  cd "$PROJECT_ROOT"

  # Run tests excluding CGO packages that require hardware
  go test -race $(go list ./... | grep -v "webserver/pkg/infrastructure/processor/loader") || {
      echo "FAILED: Go unit tests"
      exit 1
  }

  echo "OK: Go unit tests passed"
  echo ""
else
  echo "[SKIPPED] Go Unit Tests"
  echo ""
fi

# Frontend Unit Tests
if [ "$SKIP_FRONTEND" = false ]; then
  echo "[2/3] Running Frontend Unit Tests..."
  echo "====================================="
  cd "$PROJECT_ROOT/webserver/web"
  
  npm run test -- --run || {
      echo "FAILED: Frontend unit tests"
      exit 1
  }
  
  echo "OK: Frontend unit tests passed"
  echo ""
else
  echo "[SKIPPED] Frontend Unit Tests"
  echo ""
fi

# C++ Unit Tests (placeholder)
if [ "$SKIP_CPP" = false ]; then
  echo "[3/3] C++ Unit Tests..."
  echo "======================"
  echo "Not implemented yet"
  echo ""
else
  echo "[SKIPPED] C++ Unit Tests"
  echo ""
fi

echo "=================================="
echo "All Unit Tests Completed"
echo "=================================="
