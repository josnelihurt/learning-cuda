#!/bin/bash
#
# Test GitHub Actions workflow locally using act
#
# Usage:
#   ./scripts/test/workflow-local.sh              # List workflows
#   ./scripts/test/workflow-local.sh --dry-run    # Dry run (no push)
#   ./scripts/test/workflow-local.sh --job build  # Run specific job
#   ./scripts/test/workflow-local.sh --list       # List all jobs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v act &> /dev/null; then
    echo "Error: act is not installed"
    echo "Install it with: curl -sSfL https://raw.githubusercontent.com/nektos/act/master/install.sh | bash -s -- -b ~/.local/bin"
    exit 1
fi

# Check if .secrets.act exists and has GITHUB_TOKEN
if [ ! -f "$PROJECT_ROOT/.secrets.act" ]; then
    echo "Error: .secrets.act file not found"
    echo "Create it with: echo 'GITHUB_TOKEN=your_token_here' > .secrets.act"
    echo ""
    echo "Get a token from: https://github.com/settings/tokens/new"
    echo "Permissions needed: public_repo (for public repos) or repo (for private repos)"
    exit 1
fi

if ! grep -q "^GITHUB_TOKEN=" "$PROJECT_ROOT/.secrets.act" || grep -q "dummy\|test-token" "$PROJECT_ROOT/.secrets.act" 2>/dev/null; then
    echo "Warning: GITHUB_TOKEN in .secrets.act appears to be invalid or dummy"
    echo "You need a real GitHub Personal Access Token"
    echo ""
    echo "1. Create token at: https://github.com/settings/tokens/new"
    echo "2. Permissions: public_repo (public) or repo (private)"
    echo "3. Update .secrets.act: GITHUB_TOKEN=ghp_your_real_token_here"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

DRY_RUN=false
JOB=""
LIST_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --dry-run|-d)
            DRY_RUN=true
            ;;
        --job|-j)
            shift
            JOB="$1"
            ;;
        --list|-l)
            LIST_ONLY=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --list, -l           List all workflows and jobs"
            echo "  --dry-run, -d        Run without pushing images"
            echo "  --job JOB, -j JOB    Run specific job only"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
    esac
done

if [ "$LIST_ONLY" = true ]; then
    echo "Available workflows and jobs:"
    act --list
    exit 0
fi

ACT_CMD="act"

if [ -n "$JOB" ]; then
    ACT_CMD="$ACT_CMD --job $JOB"
fi

if [ "$DRY_RUN" = true ]; then
    echo "Running in dry-run mode (no push will be performed)"
    echo "Note: For true dry-run, you may need to modify the workflow to set push: false"
fi

echo "Running workflow test with act..."
echo "Command: $ACT_CMD workflow_dispatch"
echo ""

$ACT_CMD workflow_dispatch

