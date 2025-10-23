#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COVERAGE_DIR="$PROJECT_ROOT/coverage"

# Flags para omitir capas específicas
SKIP_FRONTEND=false
SKIP_GOLANG=false
SKIP_CPP=false

# Parsear argumentos
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-frontend)
      SKIP_FRONTEND=true
      shift
      ;;
    --skip-golang)
      SKIP_GOLANG=true
      shift
      ;;
    --skip-cpp)
      SKIP_CPP=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --skip-frontend    Skip frontend coverage tests"
      echo "  --skip-golang      Skip Golang coverage tests"
      echo "  --skip-cpp         Skip C++ coverage tests"
      echo ""
      echo "Examples:"
      echo "  $0                      # Run all tests"
      echo "  $0 --skip-frontend      # Skip frontend only"
      echo "  $0 --skip-frontend --skip-cpp  # Run Golang only"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "=================================="
echo "Running Coverage Tests"
echo "=================================="
echo ""

mkdir -p "$COVERAGE_DIR"/{frontend,golang,cpp}

if [ "$SKIP_FRONTEND" = false ]; then
    echo "[1/3] Running Frontend Tests with Coverage..."
    echo "================================================"
    cd "$PROJECT_ROOT/webserver/web"
    if [ -d "node_modules" ]; then
        npm run test:coverage
        echo "OK: Frontend coverage complete"
        echo "   Report: $COVERAGE_DIR/frontend/index.html"
    else
        echo "WARNING: Skipping frontend - dependencies not installed (run: cd webserver/web && npm install)"
    fi
else
    echo "[SKIPPED] Frontend Tests"
fi
echo ""

if [ "$SKIP_GOLANG" = false ]; then
    echo "[2/3] Running Golang Tests with Coverage..."
    echo "==============================================="
    cd "$PROJECT_ROOT"
    go test -v -coverprofile="$COVERAGE_DIR/golang/coverage.out" -covermode=atomic ./webserver/...
    if [ $? -eq 0 ]; then
        # Vista tradicional
        go tool cover -html="$COVERAGE_DIR/golang/coverage.out" -o "$COVERAGE_DIR/golang/traditional.html"
        go tool cover -func="$COVERAGE_DIR/golang/coverage.out" > "$COVERAGE_DIR/golang/coverage-summary.txt"
        
        # Vista treemap
        if command -v go-cover-treemap &> /dev/null; then
            go-cover-treemap -coverprofile "$COVERAGE_DIR/golang/coverage.out" > "$COVERAGE_DIR/golang/treemap.html"
            echo "OK: Treemap generated"
        else
            echo "WARNING: go-cover-treemap not installed. Install with:"
            echo "  go install github.com/nikolaydubina/go-cover-treemap@latest"
        fi
        
        echo "OK: Golang coverage complete"
        echo "   Traditional Report: $COVERAGE_DIR/golang/traditional.html"
        echo "   Treemap Report: $COVERAGE_DIR/golang/treemap.html"
        echo "   Landing Page: $COVERAGE_DIR/golang/index.html"
        echo ""
        echo "Coverage Summary:"
        tail -n 1 "$COVERAGE_DIR/golang/coverage-summary.txt"
    else
        echo "FAILED: Golang tests failed"
    fi
else
    echo "[SKIPPED] Golang Tests"
fi
echo ""

if [ "$SKIP_CPP" = false ]; then
    echo "[3/3] Running C++ Tests with Coverage..."
    echo "============================================"
    cd "$PROJECT_ROOT"
    bazel coverage //cpp_accelerator/... --combined_report=lcov 2>&1 | tee "$COVERAGE_DIR/cpp/bazel-coverage.log" || true
    if [ -f "bazel-out/_coverage/_coverage_report.dat" ]; then
        cp bazel-out/_coverage/_coverage_report.dat "$COVERAGE_DIR/cpp/coverage.lcov"
        
        if command -v genhtml &> /dev/null; then
            genhtml "$COVERAGE_DIR/cpp/coverage.lcov" -o "$COVERAGE_DIR/cpp/html" --ignore-errors source
            echo "OK: C++ coverage complete"
            echo "   Report: $COVERAGE_DIR/cpp/html/index.html"
        else
            echo "OK: C++ coverage complete (LCOV file generated)"
            echo "   LCOV: $COVERAGE_DIR/cpp/coverage.lcov"
            echo "   INFO: Install lcov for HTML reports: sudo apt-get install lcov"
        fi
    else
        echo "WARNING: C++ coverage file not found"
    fi
else
    echo "[SKIPPED] C++ Tests"
fi
echo ""

echo "=================================="
echo "Coverage Collection Complete"
echo "=================================="
echo ""
echo "View reports:"
if [ "$SKIP_FRONTEND" = false ]; then
    echo "  - Frontend: file://$COVERAGE_DIR/frontend/index.html"
fi
if [ "$SKIP_GOLANG" = false ]; then
    echo "  - Golang:   file://$COVERAGE_DIR/golang/index.html (landing page)"
    echo "    - Traditional: file://$COVERAGE_DIR/golang/traditional.html"
    echo "    - Treemap:     file://$COVERAGE_DIR/golang/treemap.html"
fi
if [ "$SKIP_CPP" = false ]; then
    echo "  - C++:      file://$COVERAGE_DIR/cpp/html/index.html"
fi
echo ""
echo "To serve all reports via HTTP:"
echo "  docker-compose -f docker-compose.dev.yml --profile coverage up coverage-report-viewer"
echo "  Then visit: http://localhost:5052"
echo ""
echo "Usage examples:"
echo "  $0                      # Run all tests"
echo "  $0 --skip-frontend      # Skip frontend only"
echo "  $0 --skip-frontend --skip-cpp  # Run Golang only"
echo "  $0 --help               # Show help"
echo ""


