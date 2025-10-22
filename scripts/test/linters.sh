#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FIX_MODE=false
if [ "$1" = "--fix" ]; then
    FIX_MODE=true
    echo "Running linters in FIX mode"
fi

echo "=================================="
echo "Running Code Linters"
echo "=================================="
echo ""

FRONTEND_ERRORS=0
GOLANG_ERRORS=0
CPP_ERRORS=0

echo "[1/3] Running Frontend Linter (ESLint)..."
echo "============================================="
cd "$PROJECT_ROOT/webserver/web"
if [ -d "node_modules" ]; then
    if [ "$FIX_MODE" = true ]; then
        npm run lint:fix || FRONTEND_ERRORS=$?
        npm run format || true
    else
        npm run lint || FRONTEND_ERRORS=$?
        npm run format:check || true
    fi
    if [ $FRONTEND_ERRORS -eq 0 ]; then
            echo "OK: Frontend linting passed"
    else
            echo "FAILED: Frontend linting found issues"
    fi
else
        echo "WARNING: Skipping frontend - dependencies not installed (run: cd webserver/web && npm install)"
fi
echo ""

echo "[2/3] Running Golang Linter (golangci-lint)..."
echo "=================================================="
cd "$PROJECT_ROOT"
if command -v golangci-lint &> /dev/null; then
    if [ "$FIX_MODE" = true ]; then
        golangci-lint run --fix ./... || GOLANG_ERRORS=$?
        go fmt ./... || true
    else
        golangci-lint run ./... || GOLANG_ERRORS=$?
    fi
    if [ $GOLANG_ERRORS -eq 0 ]; then
            echo "OK: Golang linting passed"
    else
            echo "FAILED: Golang linting found issues"
    fi
else
        echo "WARNING: golangci-lint not installed"
    echo "   Install: https://golangci-lint.run/usage/install/"
    echo "   Or use Docker: docker-compose -f docker-compose.dev.yml --profile lint up lint-golang"
fi
echo ""

echo "[3/3] Running C++ Linter (clang-tidy)..."
echo "==========================================="
cd "$PROJECT_ROOT"
if command -v clang-tidy &> /dev/null; then
    CPP_FILES=$(find cpp_accelerator -name "*.cpp" -o -name "*.h" | grep -v "BUILD" || true)
    
    if [ -n "$CPP_FILES" ]; then
        if [ "$FIX_MODE" = true ]; then
            echo "$CPP_FILES" | xargs clang-tidy -p . --fix || CPP_ERRORS=$?
            find cpp_accelerator -name "*.cpp" -o -name "*.h" | xargs clang-format -i || true
        else
            echo "$CPP_FILES" | xargs clang-tidy -p . || CPP_ERRORS=$?
            find cpp_accelerator -name "*.cpp" -o -name "*.h" | xargs clang-format --dry-run -Werror || true
        fi
        
        if [ $CPP_ERRORS -eq 0 ]; then
                echo "OK: C++ linting passed"
        else
                echo "FAILED: C++ linting found issues"
        fi
    else
            echo "WARNING: No C++ files found"
    fi
else
        echo "WARNING: clang-tidy not installed"
    echo "   Install: sudo apt-get install clang-tidy clang-format"
    echo "   Or use Docker: docker-compose -f docker-compose.dev.yml --profile lint up lint-cpp"
fi
echo ""

echo "=================================="
echo "Linting Complete"
echo "=================================="
echo ""

TOTAL_ERRORS=$((FRONTEND_ERRORS + GOLANG_ERRORS + CPP_ERRORS))

if [ $TOTAL_ERRORS -eq 0 ]; then
        echo "OK: All linters passed!"
    exit 0
else
        echo "FAILED: Some linters found issues"
    echo "   Frontend errors: $FRONTEND_ERRORS"
    echo "   Golang errors: $GOLANG_ERRORS"
    echo "   C++ errors: $CPP_ERRORS"
    echo ""
    echo "Run with --fix to auto-fix issues:"
    echo "   ./scripts/run-linters.sh --fix"
    exit 1
fi


