#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COVERAGE_DIR="$PROJECT_ROOT/coverage"

echo "=================================="
echo "Running Coverage Tests"
echo "=================================="
echo ""

mkdir -p "$COVERAGE_DIR"/{frontend,golang,cpp}

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
echo ""

echo "[2/3] Running Golang Tests with Coverage..."
echo "==============================================="
cd "$PROJECT_ROOT"
go test -v -coverprofile="$COVERAGE_DIR/golang/coverage.out" -covermode=atomic ./webserver/...
if [ $? -eq 0 ]; then
    go tool cover -html="$COVERAGE_DIR/golang/coverage.out" -o "$COVERAGE_DIR/golang/index.html"
    go tool cover -func="$COVERAGE_DIR/golang/coverage.out" > "$COVERAGE_DIR/golang/coverage-summary.txt"
        echo "OK: Golang coverage complete"
    echo "   Report: $COVERAGE_DIR/golang/index.html"
    echo ""
    echo "Coverage Summary:"
    tail -n 1 "$COVERAGE_DIR/golang/coverage-summary.txt"
else
        echo "FAILED: Golang tests failed"
fi
echo ""

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
echo ""

echo "=================================="
echo "Coverage Collection Complete"
echo "=================================="
echo ""
echo "View reports:"
echo "  - Frontend: file://$COVERAGE_DIR/frontend/index.html"
echo "  - Golang:   file://$COVERAGE_DIR/golang/index.html"
echo "  - C++:      file://$COVERAGE_DIR/cpp/html/index.html"
echo ""
echo "To serve all reports via HTTP:"
echo "  docker-compose -f docker-compose.dev.yml --profile coverage up coverage-report-viewer"
echo "  Then visit: http://localhost:5052"
echo ""


