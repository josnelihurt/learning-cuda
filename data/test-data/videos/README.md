# E2E Test Videos

This directory contains optimized videos for E2E testing purposes.

## `e2e-test.mp4`

**Purpose:** Optimized video for automated E2E and integration tests.

**Specifications:**
- **Resolution:** 480x360 (4:3 aspect ratio)
- **Duration:** 20 seconds
- **Frame Rate:** 10 fps
- **Total Frames:** 200
- **Codec:** H.264, CRF 28
- **Size:** ~464KB
- **Audio:** None (removed for smaller size)

**Source:**
- Extracted from `sample.mp4` (Big Buck Bunny)
- Start time: 288 seconds (middle section, avoiding fades)
- End time: 308 seconds

**Frame Metadata:**
- All 200 frames have been pre-extracted to `data/test-data/video-frames/e2e-test/`
- SHA256 hashes for each frame are embedded in `webserver/pkg/infrastructure/video/test_video_metadata.go`
- Frame IDs are sequential integers from 0 to 199

## Frame Extraction (On-Demand)

**Important:** The 200 PNG frames are NOT committed to the repository (they occupy 62MB).
Frames are generated on-demand when needed.

### Automatic Generation

The metadata generation tool will automatically extract frames if they don't exist:

```bash
go run cmd/generate-video-metadata/main.go
```

This command will:
1. Check if frames exist in `data/test-data/video-frames/e2e-test/`
2. If not found, automatically run `./scripts/tools/extract-frames.sh`
3. Extract 200 frames as PNGs (~62MB total)
4. Generate metadata file with SHA256 hashes

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
1. Extract a 20-second clip from the middle of `sample.mp4`
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

### Integration Tests (Go)
```go
video, err := videoRepo.GetByID(ctx, "e2e-test")
// Video will be at data/videos/e2e-test.mp4
```

### Frame Validation
```go
import "github.com/jrb/cuda-learning/webserver/pkg/infrastructure/video"

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
- 200 PNG frames = 62MB (too large for Git)
- Generated on-demand in ~2 seconds
- Scripts handle generation automatically
- Developers only need the source video (464KB)

## Notes

- Do NOT commit large videos (>1MB) to this directory
- Do NOT commit extracted PNG frames (auto-generated)
- The preview image is auto-generated on first run
- Frame extraction takes ~2 seconds for 200 frames
- Metadata generation adds ~100KB to compiled binary
- First-time setup: `go run cmd/generate-video-metadata/main.go`

