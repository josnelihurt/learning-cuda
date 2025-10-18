#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"
GITHOOKS_DIR="$SCRIPT_DIR/githooks"

echo "Installing git hooks..."

# Install pre-commit hook
if [ -f "$GITHOOKS_DIR/pre-commit.sh" ]; then
    ln -sf "../../scripts/githooks/pre-commit.sh" "$HOOKS_DIR/pre-commit"
    chmod +x "$GITHOOKS_DIR/pre-commit.sh"
    echo "Installed: pre-commit -> pre-commit.sh"
else
    echo "WARNING: pre-commit.sh not found"
fi

# Install pre-push hook
if [ -f "$GITHOOKS_DIR/pre-push" ]; then
    ln -sf "../../scripts/githooks/pre-push" "$HOOKS_DIR/pre-push"
    chmod +x "$GITHOOKS_DIR/pre-push"
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

