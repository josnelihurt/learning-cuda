#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="cuda-learning-lint-language"
DOCKERFILE_PATH="scripts/linters/language.dockerfile"

cd "$PROJECT_ROOT"

if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Building language linter Docker image..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" . || {
        echo "FAILED: Could not build Docker image"
        exit 1
    }
    echo "Docker image built successfully"
fi

echo "Language and Emoji Linter (Docker)..."

# Get staged files in host, not in container
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR | \
    grep -E '\.(cpp|h|cu|go|ts|tsx|js|jsx|proto|md|txt|sh|bash|py|java|c|cc|cxx|hpp|hxx|rs|rb|php|swift|kt|scala|lua|vim|el|pl|r|sql|html|css|xml|json|yaml|yml|toml|ini|cfg|conf|log|csv|rst|adoc|tex|srt|vtt|gitignore|dockerignore|editorconfig|env|properties)$' | \
    grep -v 'node_modules/' | \
    grep -v '/gen/' | \
    grep -v 'third_party/' | \
    grep -v '.gen.' | \
    grep -v 'package-lock.json' | \
    grep -v 'yarn.lock' | \
    grep -v 'go.sum' | \
    grep -v '.min.js' | \
    grep -v '.min.css' | \
    grep -v 'scripts/linters/language-check.sh' || true)

if [ -z "$STAGED_FILES" ]; then
    echo "No source files to check"
    exit 0
fi

# Pass files as environment variable
docker run --rm \
    -v "$PROJECT_ROOT:/workspace" \
    -v "$PROJECT_ROOT/.git:/workspace/.git:ro" \
    -w /workspace \
    -e "STAGED_FILES=$STAGED_FILES" \
    "$IMAGE_NAME" || {
    echo "FAILED: Language/Emoji linter"
    exit 1
}

echo "Language and Emoji linter passed"

