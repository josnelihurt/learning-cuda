#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_VIDEO="$PROJECT_ROOT/data/test-data/videos/e2e-test.mp4"
OUTPUT_DIR="$PROJECT_ROOT/data/test-data/video-frames/e2e-test"

echo "Extracting frames from E2E test video..."
echo "Source: $SOURCE_VIDEO"
echo "Output: $OUTPUT_DIR"

if [ ! -f "$SOURCE_VIDEO" ]; then
    echo "Error: Source video not found at $SOURCE_VIDEO"
    echo "Please run ./scripts/generate-test-video.sh first"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

EXISTING_FRAMES=$(ls -1 "$OUTPUT_DIR"/frame_*.png 2>/dev/null | wc -l)
if [ "$EXISTING_FRAMES" -ge 200 ]; then
    echo "Frames already exist ($EXISTING_FRAMES frames found)"
    echo "Skipping extraction. To regenerate, delete: $OUTPUT_DIR"
    exit 0
fi

echo ""
echo "Extracting 200 frames as PNG..."
ffmpeg -i "$SOURCE_VIDEO" \
    -vf "fps=10" \
    "$OUTPUT_DIR/frame_%04d.png"

FRAME_COUNT=$(ls -1 "$OUTPUT_DIR"/frame_*.png 2>/dev/null | wc -l)

echo ""
echo "Extraction complete!"
echo "Total frames extracted: $FRAME_COUNT"
echo "Location: $OUTPUT_DIR"
echo ""
echo "Sample frames:"
ls -lh "$OUTPUT_DIR" | head -10

