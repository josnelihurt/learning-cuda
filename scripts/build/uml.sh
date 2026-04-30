#!/bin/bash
# Generate Mermaid class diagrams for cpp_accelerator using clang-uml.
#
# Usage:
#   scripts/build/uml.sh                         # all diagrams
#   scripts/build/uml.sh --diagram cpp_domain_layer   # single diagram
#   scripts/build/uml.sh --refresh-db            # refresh compile_commands first
#
# Output: docs/uml/generated/<diagram-name>.mmd
#
# One-time setup:
#   scripts/dev/install-clang-uml.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Parse arguments ────────────────────────────────────────────────────────
DIAGRAM_FILTER=""
REFRESH_DB=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --diagram)
            DIAGRAM_FILTER="$2"
            shift 2
            ;;
        --refresh-db)
            REFRESH_DB=true
            shift
            ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# ── Prerequisites check ────────────────────────────────────────────────────
if ! command -v clang-uml &>/dev/null; then
    echo "ERROR: clang-uml not found."
    echo "       Run: scripts/dev/install-clang-uml.sh"
    exit 1
fi

if [[ ! -f compile_commands.json ]]; then
    echo "compile_commands.json not found — refreshing now..."
    REFRESH_DB=true
fi

if $REFRESH_DB; then
    echo "Refreshing compile_commands.json..."
    bazel run @hedron_compile_commands//:refresh_all
fi

# ── Generate diagrams ──────────────────────────────────────────────────────
mkdir -p docs/uml/generated

CLANG_UML_ARGS=(--generator mermaid)

if [[ -n "$DIAGRAM_FILTER" ]]; then
    CLANG_UML_ARGS+=(-n "$DIAGRAM_FILTER")
    echo "Generating diagram: $DIAGRAM_FILTER"
else
    echo "Generating all diagrams..."
fi

clang-uml "${CLANG_UML_ARGS[@]}"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "Generated diagrams in docs/uml/generated/:"
for f in docs/uml/generated/*.mmd; do
    [[ -f "$f" ]] || continue
    lines=$(wc -l < "$f")
    echo "  $(basename "$f")  (${lines} lines)"
done
echo ""
echo "View in VS Code with the Mermaid Preview extension, or paste at https://mermaid.live"
