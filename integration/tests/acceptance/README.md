# BDD Acceptance Tests for Feature Flags

## Overview

This directory contains BDD-style acceptance tests that validate the behavior of the feature flag system. These tests ensure that the `GetStreamConfig` and `SyncFeatureFlags` endpoints work correctly with Flipt.

## Prerequisites

Before running these tests, ensure the following services are running:

1. **Flipt** - Feature flag service at `http://localhost:8081`
2. **Go Service** - Application server at `https://localhost:8443`
3. **Infrastructure** - Jaeger, OTel Collector (optional but recommended)

The easiest way to start all required services is:

```bash
./scripts/start-dev.sh
```

## Running Tests

### Run all acceptance tests

From the project root:

```bash
go test ./integration/tests/acceptance/... -v
```

### Run with longer timeout

```bash
go test ./integration/tests/acceptance/... -v -timeout 30s
```

### Run specific test

```bash
go test ./integration/tests/acceptance/... -v -run TestFeatureFlagsAcceptance/GetStreamConfig
```

### Skip in short mode

These tests are skipped when running with `-short` flag:

```bash
go test -short ./integration/tests/acceptance/...  # Will skip these tests
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

These tests reuse production code from `webserver/internal/infrastructure/featureflags`:
- `FliptHTTPAPI` - Extended FliptWriter with GET methods for listing and retrieving flags
- No code duplication, tests use the same infrastructure as the application

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

## Next Steps

After these tests pass (baseline validation), you can proceed with the architectural refactoring. Re-run these tests after refactoring to ensure nothing broke.

## Future Improvements

- [ ] Integrate tests into Bazel build system
- [ ] Consider migrating to godog framework for Gherkin syntax
- [ ] Add tests for Flipt service unavailability scenarios
- [ ] Add CI/CD pipeline integration
- [ ] Add performance/load tests for flag evaluation

