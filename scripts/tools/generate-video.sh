#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SOURCE_VIDEO="$PROJECT_ROOT/data/videos/sample.mp4"
OUTPUT_VIDEO="$PROJECT_ROOT/data/test-data/videos/e2e-test.mp4"
OUTPUT_DIR="$(dirname "$OUTPUT_VIDEO")"

START_TIME=288
DURATION=20

echo "Generating optimized E2E test video..."
echo "Source: $SOURCE_VIDEO"
echo "Output: $OUTPUT_VIDEO"
echo "Start time: ${START_TIME}s (middle section, avoiding fades)"
echo "Duration: ${DURATION}s"
echo "Target specs: 480x360, 10fps, H.264"

if [ ! -f "$SOURCE_VIDEO" ]; then
    echo "Error: Source video not found at $SOURCE_VIDEO"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

ffmpeg -y \
    -ss $START_TIME \
    -i "$SOURCE_VIDEO" \
    -t $DURATION \
    -vf "scale=480:360,fps=10" \
    -c:v libx264 \
    -preset slow \
    -crf 28 \
    -an \
    "$OUTPUT_VIDEO"

echo ""
echo "Video generated successfully!"
echo ""
echo "Video information:"
ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1 "$OUTPUT_VIDEO"

FILE_SIZE=$(du -h "$OUTPUT_VIDEO" | cut -f1)
echo ""
echo "File size: $FILE_SIZE"
echo ""
echo "Please review the generated video at: $OUTPUT_VIDEO"
echo "Use: vlc $OUTPUT_VIDEO"

