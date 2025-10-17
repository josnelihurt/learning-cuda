# BDD Acceptance Tests

## Overview

This directory contains BDD-style acceptance tests that validate the behavior of the CUDA image processing service, including:
- Feature flag management with Flipt
- Image processing via ConnectRPC
- WebSocket real-time processing
- Streaming service endpoints

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

1. **Flipt** - Feature flag service at `http://localhost:8081`
2. **Go Service** - Application server at `https://localhost:8443`
3. **Infrastructure** - Jaeger, OTel Collector (optional but recommended)

The easiest way to start all required services is:

```bash
./scripts/start-dev.sh
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

## Test Scenarios

### 1. GetStreamConfig returns default values when Flipt is clean

**Given:**
- Flipt has no feature flags configured
- Service is running

**When:**
- Client calls `GetStreamConfig` endpoint

**Then:**
- Response contains default transport format: "json"
- Response contains default endpoint: "/ws"

### 2. Sync creates flags in Flipt with correct values

**Given:**
- Flipt is clean (no flags)
- Service is running

**When:**
- Client calls `SyncFeatureFlags` endpoint

**Then:**
- Flipt contains `enable_stream_transport_format` flag
- Flipt contains `enable_observability` flag
- Flag `enable_observability` has value `true`

### 3. GetStreamConfig uses Flipt values when flags exist

**Given:**
- Feature flags are already synced to Flipt
- Service is running

**When:**
- Client calls `GetStreamConfig` endpoint

**Then:**
- Response contains values from Flipt (not hardcoded defaults)

### 4. GetStreamConfig falls back to defaults when flag evaluation fails

**Given:**
- Flipt is clean (flags don't exist)
- Service is running

**When:**
- Client calls `GetStreamConfig` endpoint

**Then:**
- Response contains fallback values from config.yaml

## Test Structure

The tests follow the **Given/When/Then** BDD pattern:

### Given (Setup)
- `GivenFliptIsClean()` - Deletes all flags from Flipt
- `GivenTheServiceIsRunning()` - Verifies service health
- `GivenConfigHasDefaultValues()` - Sets expected default values

### When (Action)
- `WhenICallGetStreamConfig()` - Calls the GetStreamConfig endpoint
- `WhenICallSyncFeatureFlags()` - Calls the SyncFeatureFlags endpoint
- `WhenIWaitForFlagsToBeSynced()` - Waits for async operations

### Then (Assertion)
- `ThenTheResponseShouldContainTransportFormat()` - Validates transport format
- `ThenTheResponseShouldContainEndpoint()` - Validates endpoint
- `ThenFliptShouldHaveFlag()` - Verifies flag exists in Flipt
- `ThenFliptShouldHaveFlagWithValue()` - Verifies flag value in Flipt

## Files

- `feature_flags_test.go` - Main test scenarios
- `bdd_helpers.go` - Given/When/Then helper functions (uses FliptHTTPAPI from infrastructure)
- `README.md` - This file

## Architecture

These tests reuse production code from `webserver/pkg/infrastructure/featureflags`:
- `FliptHTTPAPI` - Extended FliptWriter with GET methods for listing and retrieving flags
- No code duplication, tests use the same infrastructure as the application

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

**Solution:** Start the service with `./scripts/start-dev.sh`

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
./scripts/start-dev.sh

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

