#!/bin/bash
set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <module_path> <version_file_path>"
    echo "Example: $0 proto proto/VERSION"
    exit 1
fi

MODULE_PATH="$1"
VERSION_FILE="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR)

MODULE_CHANGED=$(echo "$STAGED_FILES" | grep -E "^${MODULE_PATH}/" | grep -v "^${VERSION_FILE}$" || true)

if [ -z "$MODULE_CHANGED" ]; then
    exit 0
fi

VERSION_STAGED=$(echo "$STAGED_FILES" | grep -E "^${VERSION_FILE}$" || true)

if [ -z "$VERSION_STAGED" ]; then
    echo "ERROR: VERSION file ($VERSION_FILE) must be updated when files in ${MODULE_PATH}/ are modified"
    exit 1
fi

VERSION_DIFF=$(git diff --cached "$VERSION_FILE" || true)
if [ -z "$VERSION_DIFF" ]; then
    echo "ERROR: VERSION file ($VERSION_FILE) is staged but content has not changed"
    exit 1
fi

exit 0
