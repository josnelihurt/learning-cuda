#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "Installing git hooks..."

# Install pre-commit hook
if [ -f "$SCRIPT_DIR/pre-commit.sh" ]; then
    ln -sf "../../scripts/hooks/pre-commit.sh" "$HOOKS_DIR/pre-commit"
    chmod +x "$SCRIPT_DIR/pre-commit.sh"
    echo "Installed: pre-commit -> pre-commit.sh"
else
    echo "WARNING: pre-commit.sh not found"
fi

# Install pre-push hook
if [ -f "$SCRIPT_DIR/pre-push" ]; then
    ln -sf "../../scripts/hooks/pre-push" "$HOOKS_DIR/pre-push"
    chmod +x "$SCRIPT_DIR/pre-push"
    echo "Installed: pre-push"
else
    echo "WARNING: pre-push not found"
fi

echo ""
echo "Git hooks installed"
echo ""
echo "Hooks:"
echo "  pre-commit: Unit tests + linters"
echo "  pre-push: Full validation (all browsers)"
echo ""
echo "Skip when needed: git commit --no-verify"

