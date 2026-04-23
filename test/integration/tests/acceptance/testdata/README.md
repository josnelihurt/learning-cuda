# Test Data Directory

This directory is a placeholder for acceptance test data files.

## Current State

This directory is currently empty. All test data used by the acceptance tests is generated programmatically at runtime or referenced from the project's `data/` directory.

## Test Data Sources

### Static Images

Static images served by the service are loaded from `data/static_images/`:
- `airplane.png`, `barbara.png`, `cameraman.png`, `couple.png`, `goldhill.png`
- `house.png`, `lena.png`, `mandrill.png`, `peppers.png`, `sailboat.png`

These are validated by the `available_images.feature` test scenarios.

### Videos

Video files served by the service are loaded from `data/videos/`:
- `sample.mp4`, `test-small.mp4`, `e2e-test.mp4`

### Video Previews

Preview thumbnails for videos are stored in `data/video_previews/`:
- Generated automatically when videos are uploaded
- Named as `{video_id}.png`

### Models

YOLO detection models are stored in `data/models/`:
- `yolov10n.onnx`

### Programmatically Generated Test Data

The following test data is created at runtime by the test step definitions (see `steps/bdd_context.go`):

- **PNG images** for upload tests: Generated via `createTestPNGImage()` (100x100 minimal valid PNG)
- **MP4 videos** for upload tests: Generated via `createTestMP4Video()` (minimal valid ftyp+mdat structure)
- **Large file payloads**: Allocated in memory to test size limit validation (11MB for images, 101MB for videos)
- **Invalid format payloads**: Small byte arrays with incorrect magic bytes (JPEG header for PNG tests, AVI header for MP4 tests)

### External Test Data

Some video upload tests read real video files from `data/videos/test-small.mp4` when available (e.g., the `preview-test.mp4` upload scenario).

## Adding Test Data Files

If checksum files or other static test data are needed in the future, place them in this directory and update this README accordingly.
