#!/bin/bash
set -e

# Get the actual script location (follows symlinks)
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Running pre-commit validations..."
echo ""

# Get all staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR)

# Detect changes by component
CPP_CHANGED=$(echo "$STAGED_FILES" | grep -E '^cpp_accelerator/|^third_party/|^MODULE\.bazel|^WORKSPACE' || true)
GO_CHANGED=$(echo "$STAGED_FILES" | grep -E '^webserver/.*\.go$|^proto/|^go\.(mod|sum)$' || true)
FRONTEND_CHANGED=$(echo "$STAGED_FILES" | grep -E '^webserver/web/' || true)

# Run C++ tests and linting only if changes detected
if [ -n "$CPP_CHANGED" ]; then
    "$SCRIPT_DIR/pre-commit-cpp.sh" || exit 1
    echo ""
    "$SCRIPT_DIR/pre-commit-lint-cpp.sh" || exit 1
    echo ""
else
    echo "Skipping C++ tests and linting (no changes detected)"
    echo ""
fi

# Run Go tests and linting only if changes detected
if [ -n "$GO_CHANGED" ]; then
    "$SCRIPT_DIR/pre-commit-go.sh" || exit 1
    echo ""
    "$SCRIPT_DIR/pre-commit-lint-go.sh" || exit 1
    echo ""
else
    echo "Skipping Go tests and linting (no changes detected)"
    echo ""
fi

# Run Frontend tests and linting only if changes detected
if [ -n "$FRONTEND_CHANGED" ]; then
    "$SCRIPT_DIR/pre-commit-frontend.sh" || exit 1
    echo ""
    "$SCRIPT_DIR/pre-commit-lint-frontend.sh" || exit 1
else
    echo "Skipping Frontend tests and linting (no changes detected)"
fi

echo ""
"$SCRIPT_DIR/pre-commit-lint-language.sh" || exit 1

cd "$PROJECT_ROOT"
echo ""
echo "Pre-commit validations passed"

