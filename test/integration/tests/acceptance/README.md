# BDD Acceptance Tests

## Overview

This directory contains BDD-style acceptance tests that validate the behavior of the CUDA image processing service, including:
- Image processing via ConnectRPC (Grayscale, Gaussian Blur, Multi-filter)
- WebSocket real-time processing
- Input source management (static images, videos, camera)
- Video playback and frame ID tracking
- Processor capabilities and system information
- Tools configuration

## Prerequisites

Before running these tests, ensure the following:

### 1. Proto Files Generated

If you've modified `proto/image_processing.proto`, regenerate the proto files:

```bash
# Option 1: Run setup script (checks and regenerates if needed)
./integration/tests/acceptance/scripts/setup.sh

# Option 2: Manually regenerate
docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
```

### 2. Services Running

Ensure the following services are running:

1. **Go Service** - Application server at `https://localhost:8443`
2. **Infrastructure** - Jaeger, OTel Collector (optional but recommended)

The easiest way to start all required services is:

```bash
./scripts/dev/start.sh
```

## Running Tests

### BDD Tests with Godog/Gherkin (Recommended)

From the project root:

```bash
go test ./integration/tests/acceptance -run TestFeatures -v
```

### Generate Reports

**Cucumber JSON** (for Allure and other tools):
```bash
go test ./integration/tests/acceptance -run TestFeatures -v \
  -godog.format=cucumber \
  -godog.output=cucumber-report.json
```

**JUnit XML** (for CI/CD):
```bash
go test ./integration/tests/acceptance -run TestFeatures -v \
  -godog.format=junit > junit-report.xml
```

**Multiple formats at once**:
```bash
go test ./integration/tests/acceptance -run TestFeatures -v \
  -godog.format=pretty,cucumber:cucumber-report.json,junit:junit-report.xml
```

### Legacy Tests (Deprecated)

The original Go-native tests still work:
```bash
go test ./integration/tests/acceptance -run TestFeatureFlagsAcceptance -v
```

### Skip in short mode

Tests are skipped when running with `-short` flag:
```bash
go test -short ./integration/tests/acceptance  # Will skip these tests
```

## Test Features

The test suite includes 11 feature files covering:

### Image Processing Features
- **`features/image_processing.feature`** (22 scenarios)
  - Passthrough processing (no filters)
  - Grayscale filters (5 algorithms: BT601, BT709, Average, Lightness, Luminosity)
  - Gaussian Blur filters (various kernel sizes, sigma values, border modes)
  - Multi-filter combinations (Grayscale + Blur, Blur + Grayscale)
  - Error scenarios (invalid filters, missing parameters)

- **`features/websocket_processing.feature`** (9 scenarios)
  - Real-time image processing via WebSocket
  - Frame-by-frame video processing
  - Error handling for malformed requests

### Input Source Features
- **`features/input_sources.feature`** (3 scenarios)
  - Listing available input sources (static, camera, video)

- **`features/available_images.feature`** (3 scenarios)
  - Listing static images available for processing

- **`features/upload_images.feature`** (4 scenarios)
  - Uploading PNG images with validation

- **`features/video_playback.feature`** (9 scenarios)
  - Listing available videos
  - Uploading MP4 videos
  - Video metadata validation

- **`features/video_frame_id.feature`** (6 scenarios)
  - Sequential frame ID tracking during video processing

### System Features
- **`features/processor_capabilities.feature`** (5 scenarios)
  - Querying available filters and their parameters
  - Supported accelerators (CUDA, CPU)
  - Filter parameter types and constraints

- **`features/system_info.feature`** (4 scenarios)
  - Retrieving system version information
  - Environment configuration details

- **`features/tools_configuration.feature`** (6 scenarios)
  - Dynamic tool discovery (Jaeger, Grafana, etc.)
  - Tool configuration and accessibility

- **`features/stream_config.feature`** (3 scenarios)
  - Stream configuration retrieval
  - Transport format settings

## Test Structure

The tests follow the **Given/When/Then** BDD pattern with step definitions organized by responsibility:

### Step Definition Files
- `steps/given_steps.go` - Setup steps (service running, config values)
- `steps/when_steps.go` - Action steps (calling endpoints, processing images)
- `steps/then_steps.go` - Assertion steps (validating responses)
- `steps/image_steps.go` - Image processing and WebSocket steps
- `steps/input_source_steps.go` - Input source and available image steps
- `steps/video_steps.go` - Video playback and frame tracking steps

### Core Components
- `godog_test.go` - Test suite entry point with TestFeatures
- `steps/context.go` - TestContext wrapper for dependency injection
- `steps/bdd_context.go` - BDDContext with business logic and client management

### Test Data
- `testdata/checksums.json` - MD5 checksums for deterministic image processing validation
- `scripts/generate_checksums.go` - Tool for regenerating checksums after visual inspection

### Then (Assertion)
- `ThenTheResponseShouldContainTransportFormat()` - Validates transport format
- `ThenTheResponseShouldContainEndpoint()` - Validates endpoint
- `ThenFliptShouldHaveFlag()` - Verifies flag exists in Flipt
- `ThenFliptShouldHaveFlagWithValue()` - Verifies flag value in Flipt

## Files

- `godog_test.go` - Test suite entry point with TestFeatures
- `steps/context.go` - TestContext wrapper for dependency injection
- `steps/bdd_context.go` - BDDContext with business logic and client management
- `steps/given_steps.go` - Given step definitions
- `steps/when_steps.go` - When step definitions
- `steps/then_steps.go` - Then step definitions
- `steps/image_steps.go` - Image processing and WebSocket steps
- `steps/input_source_steps.go` - Input source and available image steps
- `steps/video_steps.go` - Video playback and frame tracking steps
- `features/` - Gherkin feature files (11 features)
- `README.md` - This file

## Architecture

These tests use ConnectRPC clients generated from Protocol Buffer definitions:
- **genconnect.ImageProcessorServiceClient** - Image processing operations
- **genconnect.ConfigServiceClient** - Configuration queries
- **genconnect.FileServiceClient** - File upload and listing operations

The tests make real gRPC-Web requests to the running service, validating end-to-end behavior including network communication, serialization, and business logic.

## Running Tests with Convenience Script

A convenience script is available that handles service checks, checksum generation, and test execution:

```bash
cd test/integration/tests/acceptance/scripts
./run_tests.sh
```

This script:
1. Verifies the service is running
2. Generates checksums automatically if needed
3. Executes tests with a 120-second timeout

## Test Combinations

### Filters
- **NONE** - Passthrough (no filter applied)
- **GRAYSCALE** - Grayscale conversion (5 algorithms available)
- **BLUR** - Gaussian blur (configurable kernel, sigma, border mode)

### Accelerators
- **CUDA** - GPU processing
- **CPU** - CPU fallback processing

### Grayscale Algorithms
- **BT601** - ITU-R BT.601 (SDTV): Y = 0.299R + 0.587G + 0.114B
- **BT709** - ITU-R BT.709 (HDTV): Y = 0.2126R + 0.7152G + 0.0722B
- **AVERAGE** - Simple average: Y = (R + G + B) / 3
- **LIGHTNESS** - Lightness: Y = (max(R,G,B) + min(R,G,B)) / 2
- **LUMINOSITY** - Luminosity: Y = 0.21R + 0.72G + 0.07B

### Blur Parameters
- **Kernel Size**: 3, 5, 7 (odd numbers)
- **Sigma**: 0.5, 1.0, 1.5, 2.0
- **Border Mode**: REFLECT, CLAMP, WRAP
- **Separable**: true/false (optimization for Gaussian blur)

### Multi-filter Combinations
- **GRAYSCALE_AND_BLUR** - Apply grayscale first, then blur
- **BLUR_AND_GRAYSCALE** - Apply blur first, then grayscale

## Adding New BDD Tests

### Workflow for Adding New Test Scenarios

#### Step 1: Define the API Contract (Protobuf)

If your feature requires a new RPC endpoint:

1. **Edit `proto/image_processing.proto`**:
   ```protobuf
   message ListInputsRequest {}
   
   message ListInputsResponse {
     repeated InputSource sources = 1 [json_name = "sources"];
   }
   
   service ConfigService {
     rpc ListInputs(ListInputsRequest) returns (ListInputsResponse);
   }
   ```

2. **Generate proto files**:
   ```bash
   ./integration/tests/acceptance/scripts/setup.sh
   # or manually:
   docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
   ```

3. **Verify generation**:
   ```bash
   # Check that types are generated
   grep -q "ListInputs" proto/gen/image_processing.pb.go && echo "✅ Proto generated"
   ```

#### Step 2: Write BDD Scenarios

1. **Create/edit feature file** (`features/your_feature.feature`):
   ```gherkin
   Scenario: New feature behavior
     Given some initial state
     When I perform an action
     Then I expect some outcome
   ```

2. **Run tests to see missing steps**:
   ```bash
   go test ./integration/tests/acceptance -run TestFeatures -v
   ```
   Godog will show you which step definitions are missing.

#### Step 3: Implement Step Definitions

1. **Create step file** (e.g., `steps/your_feature_steps.go`):
   ```go
   func InitializeYourFeatureSteps(ctx *godog.ScenarioContext, tc *TestContext) {
       ctx.Step(`^I do something$`, tc.iDoSomething)
   }
   ```

2. **Add implementation in `steps/bdd_context.go`**:
   ```go
   func (c *BDDContext) WhenIDoSomething() error {
       // Use the generated proto client
       resp, err := c.configClient.YourMethod(ctx, connect.NewRequest(&pb.YourRequest{}))
       return err
   }
   ```

3. **Register steps in `godog_test.go`**:
   ```go
   steps.InitializeYourFeatureSteps(ctx, testCtx)
   ```

#### Step 4: Verify and Iterate

1. **Re-run tests** to verify they pass:
   ```bash
   go test ./integration/tests/acceptance -run TestFeatures -v
   ```

2. **Fix any issues** and iterate until all scenarios pass.

### Example: Adding ListInputs Endpoint (Complete Walkthrough)

This is a real example from this codebase showing the complete BDD workflow:

**1. Define Proto** (`proto/image_processing.proto`):
```protobuf
message InputSource {
  string id = 1 [json_name = "id"];
  string display_name = 2 [json_name = "display_name"];
  string type = 3 [json_name = "type"];
  string image_path = 4 [json_name = "image_path"];
  bool is_default = 5 [json_name = "is_default"];
}

message ListInputsRequest {}

message ListInputsResponse {
  repeated InputSource sources = 1 [json_name = "sources"];
}

service ConfigService {
  rpc ListInputs(ListInputsRequest) returns (ListInputsResponse);
}
```

**2. Generate Proto**:
```bash
./integration/tests/acceptance/scripts/setup.sh
```

**3. Create Feature** (`features/input_sources.feature`):
```gherkin
Feature: Input Source Selection
  Scenario: List default input sources
    Given the service is running at "https://localhost:8443"
    When I call ListInputs endpoint
    Then the response should succeed
    And the response should contain input source "lena" with type "static"
```

**4. Create Steps** (`steps/input_source_steps.go`):
```go
func InitializeInputSourceSteps(ctx *godog.ScenarioContext, tc *TestContext) {
    ctx.Step(`^I call ListInputs endpoint$`, tc.iCallListInputsEndpoint)
    ctx.Step(`^the response should contain input source "([^"]*)" with type "([^"]*)"$`, 
        tc.theResponseShouldContainInputSourceWithType)
}

func (tc *TestContext) iCallListInputsEndpoint() error {
    return tc.WhenICallListInputs()
}
```

**5. Implement Logic** (`steps/bdd_context.go`):
```go
func (c *BDDContext) WhenICallListInputs() error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    resp, err := c.configClient.ListInputs(ctx, connect.NewRequest(&pb.ListInputsRequest{}))
    if err != nil {
        return fmt.Errorf("failed to call ListInputs: %w", err)
    }

    c.inputSources = resp.Msg.Sources
    return nil
}
```

**6. Register in Test Suite** (`godog_test.go`):
```go
steps.InitializeInputSourceSteps(ctx, testCtx)
```

## Troubleshooting

### Service not running

```
Error: service is not running at https://localhost:8443
```

**Solution:** Start the service with `./scripts/dev/start.sh`

### Flipt not accessible

```
Error: failed to clean Flipt: Get "http://localhost:8081/api/v1/namespaces/default/flags": dial tcp [::1]:8081: connect: connection refused
```

**Solution:** Ensure Flipt is running:
```bash
docker compose -f docker-compose.dev.yml up -d flipt
```

### Certificate errors

```
Error: x509: certificate signed by unknown authority
```

**Solution:** The tests already use `InsecureSkipVerify: true` for development. If you still see this error, check TLS configuration.

### Tests timing out

**Solution:** Increase timeout:
```bash
go test ./integration/tests/acceptance/... -v -timeout 60s
```

## Image Processing Tests

### Test Features

The suite now includes comprehensive tests for:

1. **`image_processing.feature`** - ConnectRPC Image Processing
   - 12 successful processing scenarios (filters × accelerators × grayscale types)
   - 3 error scenarios (empty image, zero dimensions, invalid channels)
   - Checksum validation for consistent output

2. **`websocket_processing.feature`** - Real-time WebSocket Processing
   - JSON and Binary transport formats
   - 4 successful frame processing scenarios
   - 2 error scenarios (empty request, empty image)
   - Checksum validation for processed frames

3. **`streaming_service.feature`** - Streaming Service Status
   - Validates StreamProcessVideo returns Unimplemented

### Generating Test Checksums

Before running image processing tests, generate checksums:

```bash
# Start the service first
./scripts/dev/start.sh

# In another terminal, generate checksums
cd integration/tests/acceptance/scripts
./run_checksum_generation.sh
```

This will process test images with all combinations and save checksums to `testdata/checksums.json`.

### Test Combinations

**Filters:**
- `FILTER_TYPE_NONE` - No processing (passthrough)
- `FILTER_TYPE_GRAYSCALE` - Convert to grayscale

**Accelerators:**
- `ACCELERATOR_TYPE_CUDA` - CUDA GPU processing
- `ACCELERATOR_TYPE_CPU` - CPU processing

**Grayscale Algorithms:**
- `GRAYSCALE_TYPE_BT601` - ITU-R BT.601 (SDTV)
- `GRAYSCALE_TYPE_BT709` - ITU-R BT.709 (HDTV)
- `GRAYSCALE_TYPE_AVERAGE` - Simple average
- `GRAYSCALE_TYPE_LIGHTNESS` - (max + min) / 2
- `GRAYSCALE_TYPE_LUMINOSITY` - Weighted average

### Checksum Validation

Tests use SHA-256 checksums to validate that:
- Image processing produces consistent results
- GPU and CPU implementations match expected outputs
- Different grayscale algorithms produce correct transformations
- WebSocket processing matches ConnectRPC results

## Next Steps

After these tests pass (baseline validation), you can proceed with the architectural refactoring. Re-run these tests after refactoring to ensure nothing broke.

## Future Improvements

- [ ] Add more test images with different formats (JPEG, etc.)
- [ ] Add tests for batch processing
- [ ] Add performance benchmarks
- [ ] Integrate tests into Bazel build system
- [ ] Add tests for Flipt service unavailability scenarios
- [ ] Add CI/CD pipeline integration
- [ ] Add performance/load tests for flag evaluation

