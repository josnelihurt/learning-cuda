# Test Data Directory

This directory contains checksums for integration tests.

## Images

Test images are loaded from the project's `data/` directory to avoid duplication:
- `../../../../data/lena.png`: Standard test image used for image processing validation

## Checksums

The `checksums.json` file contains SHA-256 checksums of processed images for each combination of:
- Image file
- Filter type (NONE, GRAYSCALE)
- Accelerator type (GPU, CPU)
- Grayscale algorithm (BT601, BT709, AVERAGE, LIGHTNESS, LUMINOSITY)

## Generating Checksums

**Prerequisites:**
1. Ensure the CUDA service is running on `https://localhost:8443`
2. Make sure the service has GPU access (or use CPU-only mode)

**To generate checksums:**

```bash
cd integration/tests/acceptance/scripts
./run_checksum_generation.sh
```

Or manually:

```bash
cd integration/tests/acceptance/scripts
go run generate_checksums.go https://localhost:8443
```

This will:
1. Load each test image
2. Process it with all filter/accelerator/grayscale combinations
3. Calculate SHA-256 checksums of the results
4. Save checksums to `testdata/checksums.json`

## Checksum Format

```json
{
  "generated_at": "2024-01-01T12:00:00Z",
  "checksums": [
    {
      "image": "lena.png",
      "filter": "FILTER_TYPE_NONE",
      "accelerator": "ACCELERATOR_TYPE_CUDA",
      "grayscale_type": "",
      "checksum": "abc123...",
      "width": 512,
      "height": 512,
      "channels": 4
    }
  ]
}
```

## Usage in Tests

Tests automatically load checksums from `checksums.json` and compare them against processed images to validate that image processing produces consistent results.

