#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "C++ Unit Tests..."
cd "$PROJECT_ROOT"
bazel test //cpp_accelerator/... --test_output=errors || {
    echo "FAILED: C++ tests"
    exit 1
}
echo "✓ C++ tests passed"

