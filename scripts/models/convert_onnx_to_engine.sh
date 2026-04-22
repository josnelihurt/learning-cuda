#!/bin/bash
# Converts ONNX models to TensorRT engine files for the current architecture.
#
# Usage:
#   ./convert_onnx_to_engine.sh <onnx_file> [output_dir]
#
# Examples:
#   ./convert_onnx_to_engine.sh yolov10n.onnx
#   ./convert_onnx_to_engine.sh data/models/yolov10n.onnx data/models/
#
# Requirements:
#   - TensorRT must be installed (trtexec command available)
#   - For Jetson: TensorRT is pre-installed with JetPack
#   - For desktop: Install via NVIDIA packages
#
# Output:
#   Generates <basename>.engine in the output directory
#
# Note: Engine files are architecture-specific. An engine built on x86_64
#       will not work on ARM64/Jetson and vice versa.

set -e

ONNX_FILE="$1"
OUTPUT_DIR="${2:-$(dirname "$ONNX_FILE")}"
BASENAME=$(basename "$ONNX_FILE" .onnx)

if [ ! -f "$ONNX_FILE" ]; then
    echo "Error: ONNX file not found: $ONNX_FILE"
    exit 1
fi

if ! command -v trtexec &> /dev/null; then
    echo "Error: trtexec not found. Please install TensorRT."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Converting $ONNX_FILE to TensorRT engine..."
echo "Output: $OUTPUT_DIR/${BASENAME}.engine"
echo "Architecture: $(uname -m)"

# Detect if running on Jetson for potential JetPack-specific optimization
if [ -f /etc/nv_tegra_release ]; then
    echo "Detected Jetson platform"
    JETPACK_VERSION=$(head -n1 /etc/nv_tegra_release | awk -F= '{print $2}' | tr -d ' ')
    echo "JetPack version: $JETPACK_VERSION"

    # Tag engine file with JetPack version for identification
    ENGINE_SUFFIX=".jp${JETPACK_VERSION%%.*}.engine"
else
    ENGINE_SUFFIX=".engine"
fi

trtexec --onnx="$ONNX_FILE" \
        --saveEngine="$OUTPUT_DIR/${BASENAME}${ENGINE_SUFFIX}" \
        --fp16 \
        --workspace=1024 \
        --verbose

if [ $? -eq 0 ]; then
    echo "✓ Engine generated successfully: $OUTPUT_DIR/${BASENAME}${ENGINE_SUFFIX}"
    ls -lh "$OUTPUT_DIR/${BASENAME}${ENGINE_SUFFIX}"
else
    echo "✗ Engine generation failed"
    exit 1
fi
