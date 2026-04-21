# E2E Test Videos

This directory contains optimized videos for E2E testing purposes.

## `e2e-test.mp4`

**Purpose:** Optimized video for automated E2E and integration tests.

**Specifications:**
- **Resolution:** 480x360 (16:9 display aspect ratio, 4:3 sample aspect ratio)
- **Duration:** 20 seconds
- **Frame Rate:** 10 fps
- **Total Frames:** 200
- **Codec:** H.264 (High profile), CRF 28
- **Size:** ~464KB (474,607 bytes)
- **Audio:** None (removed for smaller size)

**Source:**
- Extracted from `data/videos/sample.mp4` (Big Buck Bunny)
- Start time: 288 seconds (middle section, avoiding fades)
- End time: 308 seconds

**Frame Metadata:**
- 200 PNG frames can be extracted to `data/test-data/video-frames/e2e-test/` (~62MB)
- Frame filenames use 1-based numbering: `frame_0001.png` to `frame_0200.png` (ffmpeg convention)
- SHA256 hashes for each frame can be generated via the metadata tool (see below)

## Frame Extraction (On-Demand)

The 200 PNG frames are NOT committed to the repository (they occupy ~62MB and are excluded via `.gitignore`).
Frames are generated on-demand when needed.

### Automatic Generation

The metadata generation tool will automatically extract frames if they don't exist:

```bash
go run ./src/tools/generate-video-metadata
```

This command will:
1. Check if frames exist in `data/test-data/video-frames/e2e-test/`
2. If not found, automatically run `./scripts/tools/extract-frames.sh`
3. Extract 200 frames as PNGs (~62MB total)
4. Generate `src/go_api/pkg/infrastructure/video/test_video_metadata.go` with SHA256 hashes

### Manual Extraction

You can also extract frames manually:

```bash
./scripts/tools/extract-frames.sh
```

The script will:
- Check if 200 frames already exist
- Skip extraction if frames are present
- Extract only if needed

To force regeneration, delete the frames directory:

```bash
rm -rf data/test-data/video-frames/e2e-test
./scripts/tools/extract-frames.sh
```

## Regenerating Test Video

If you need to regenerate the test video:

```bash
./scripts/tools/generate-video.sh
```

This will:
1. Extract a 20-second clip from the middle of `data/videos/sample.mp4`
2. Downscale to 480x360
3. Reduce framerate to 10fps
4. Save to `data/test-data/videos/e2e-test.mp4`

## Usage in Tests

### E2E Tests (Playwright)
The video is automatically available in the video selector as "E2e test".

```typescript
await page.getByRole('button', { name: 'Videos' }).click();
await page.locator('[data-testid="video-card-e2e-test"]').click();
```

Tests are located at `src/front-end/tests/e2e/video-playback.spec.ts` and cover:
- Video preview thumbnail display
- Pixel-level frame validation (SHA-256 hash comparison)
- Frame sequential ID validation
- Grayscale filter application
- Stress testing with multiple sources and filter toggling

### Integration Tests (Go)
```go
video, err := videoRepo.GetByID(ctx, "e2e-test")
```

Note: The Go `FileVideoRepository` scans the `data/videos/` directory for `.mp4` files. Test videos in this directory (`data/test-data/videos/`) are separate and used primarily by frontend E2E tests.

### Frame Metadata (Generated)
After running `go run ./src/tools/generate-video-metadata`, frame hash validation is available:

```go
import "github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/video"

// Get metadata for a specific frame
metadata := video.GetFrameMetadata(42)
fmt.Println(metadata.Hash) // SHA256 hash of frame 42

// Validate a frame hash
isValid := video.ValidateFrameHash(42, actualHash)
```

## Why This Approach?

**Problems with large test videos:**
- Large repository size
- Slow CI/CD pipelines
- Non-deterministic pixel validation

**Benefits of optimized test video:**
- Small size (~464KB vs 150MB+)
- Deterministic frame validation with pre-calculated hashes
- Fast test execution
- Suitable for version control
- Realistic enough for image processing validation

**Why frames are NOT in repository:**
- 200 PNG frames = ~62MB (too large for Git)
- Generated on-demand in ~2 seconds
- Scripts handle generation automatically
- Developers only need the source video (464KB)

## Notes

- Do NOT commit large videos (>1MB) to this directory
- Do NOT commit extracted PNG frames (auto-generated, excluded via `.gitignore`)
- Frame extraction takes ~2 seconds for 200 frames
- The metadata file (`test_video_metadata.go`) must be generated before Go tests that use frame validation can compile
- First-time setup: `go run ./src/tools/generate-video-metadata`
