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

# ── Rebuild index.md (only when all diagrams were generated) ──────────────
if [[ -z "$DIAGRAM_FILTER" ]]; then
    INDEX="docs/uml/index.md"
    {
        echo "# UML Class Diagrams — cpp_accelerator"
        echo ""
        echo "> Generado con [clang-uml](https://github.com/bkryza/clang-uml). Para regenerar: \`scripts/build/uml.sh\`"
        echo ""

        declare -A TITLES=(
            [cpp_domain_layer]="1. Domain Layer|Interfaces y modelos puros — sin dependencias de infraestructura."
            [cpp_application_layer]="2. Application Layer|Engine, Pipeline, Filter Factory Registry."
            [cpp_ports]="3. Ports|Interfaces de los puertos hexagonales (control y media)."
            [cpp_core]="4. Core Utilities|Logger, Telemetry, Result, SignalHandler."
            [cpp_control_adapters]="5. Control & Media Adapters|gRPC outbound client y WebRTC."
            [cpp_compute_backends]="6. Compute Backends|CUDA, OpenCL, Vulkan y CPU — factories y filtros."
        )
        ORDER=(cpp_domain_layer cpp_application_layer cpp_ports cpp_core cpp_control_adapters cpp_compute_backends)

        for key in "${ORDER[@]}"; do
            IFS='|' read -r title desc <<< "${TITLES[$key]}"
            echo "---"
            echo ""
            echo "## $title"
            echo ""
            echo "$desc"
            echo ""
            echo '```mermaid'
            cat "docs/uml/generated/${key}.mmd"
            echo '```'
            echo ""
        done
    } > "$INDEX"
    echo "  index.md rebuilt  ($(wc -l < "$INDEX") lines)"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "Generated diagrams in docs/uml/generated/:"
for f in docs/uml/generated/*.mmd; do
    [[ -f "$f" ]] || continue
    lines=$(wc -l < "$f")
    echo "  $(basename "$f")  (${lines} lines)"
done
echo ""
echo "Open docs/uml/index.md in VS Code and press Ctrl+Shift+V to view all diagrams."
echo "Or paste individual .mmd files at https://mermaid.live"
