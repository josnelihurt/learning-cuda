#!/bin/bash
set -e

# Get the actual script location (follows symlinks)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Running pre-commit validations..."
echo ""

# Run each layer's tests
"$SCRIPT_DIR/pre-commit-cpp.sh" || exit 1
echo ""

"$SCRIPT_DIR/pre-commit-go.sh" || exit 1
echo ""

"$SCRIPT_DIR/pre-commit-frontend.sh" || exit 1
echo ""

# Run linters
"$SCRIPT_DIR/pre-commit-lint-go.sh" || exit 1
echo ""

"$SCRIPT_DIR/pre-commit-lint-cpp.sh" || exit 1
echo ""

"$SCRIPT_DIR/pre-commit-lint-frontend.sh" || exit 1

cd "$PROJECT_ROOT"
echo ""
echo "Pre-commit validations passed"

