# Testing & Code Quality

This document describes the testing strategy and how to execute the different test suites in the project.

## Testing Strategy

The project uses a multi-layered testing approach:

1. **Unit Tests** - Fast, isolated tests for individual components
2. **BDD Tests** - Behavior-driven acceptance tests using Gherkin
3. **E2E Tests** - End-to-end browser tests using Playwright

Each layer validates different aspects of the system and can be run independently or together.

## Test Types

### Unit Tests

Fast, isolated tests that verify individual components work correctly.

#### Frontend Unit Tests (Vitest)

**Location:** `webserver/web/`

**Technology:** Vitest + TypeScript

**What they test:**
- Web Components (Lit)
- Services and utilities
- Component interactions
- State management

**Run:**
```bash
cd webserver/web
npm run test
```

**With watch mode:**
```bash
npm run test -- --watch
```

**Via script:**
```bash
./scripts/test/unit-tests.sh --skip-golang --skip-cpp
```

#### Golang Unit Tests

**Location:** `webserver/pkg/`

**Technology:** `go test` with race detection

**What they test:**
- Use cases (application layer)
- Domain logic
- Infrastructure repositories (mocked)
- Handler logic

**Run:**
```bash
go test -race ./webserver/pkg/...
```

**Via script:**
```bash
./scripts/test/unit-tests.sh --skip-frontend --skip-cpp
```

**Exclusions:** Tests exclude CGO packages requiring hardware (`processor/loader`), integration tests, and generated proto code.

#### C++ Unit Tests

**Location:** `cpp_accelerator/`

**Technology:** GoogleTest + Bazel

**Status:** Not yet implemented

**Via script:**
```bash
./scripts/test/unit-tests.sh --skip-frontend --skip-golang
```

#### Run All Unit Tests

```bash
./scripts/test/unit-tests.sh
```

**Options:**
- `--skip-golang` - Skip Go tests
- `--skip-frontend` - Skip frontend tests
- `--skip-cpp` - Skip C++ tests
- `--help` - Show usage

### BDD Tests (Behavior-Driven Development)

Acceptance tests using Gherkin feature files and Godog.

**Location:** `integration/tests/acceptance/`

**Technology:** Godog (Go) + Gherkin

**What they test:**
- API endpoints (Connect-RPC)
- Feature flag management
- Image processing workflows
- WebSocket processing
- Input source management
- Processor capabilities
- Tools configuration

**Feature Files:**
- `image_processing.feature` - Image processing via ConnectRPC
- `websocket_processing.feature` - Real-time WebSocket processing
- `feature_flags.feature` - Feature flag operations
- `input_sources.feature` - Input source management
- `processor_capabilities.feature` - Filter discovery
- `tools_configuration.feature` - Dynamic tools
- `available_images.feature` - Image selection

#### Prerequisites

**1. Services Running:**

Required services must be running:
- Flipt (feature flags) at `http://localhost:8081`
- Go service at `https://localhost:8443`

Start services:
```bash
./scripts/dev/start.sh
```

**2. Proto Files Generated:**

If you've modified proto files:
```bash
./integration/tests/acceptance/scripts/setup.sh
```

Or manually:
```bash
docker run --rm -v $(pwd):/workspace -u $(id -u):$(id -g) cuda-learning-bufgen:latest generate
```

#### Run BDD Tests

**From project root:**
```bash
go test ./integration/tests/acceptance -run TestFeatures -v
```

**With Docker:**
```bash
./scripts/test/integration.sh backend
```

#### Generate Reports

**Cucumber JSON:**
```bash
go test ./integration/tests/acceptance -run TestFeatures -v \
  -godog.format=cucumber \
  -godog.output=cucumber-report.json
```

**JUnit XML:**
```bash
go test ./integration/tests/acceptance -run TestFeatures -v \
  -godog.format=junit > junit-report.xml
```

**Multiple formats:**
```bash
go test ./integration/tests/acceptance -run TestFeatures -v \
  -godog.format=pretty,cucumber:cucumber-report.json,junit:junit-report.xml
```

**View HTML report:**
```bash
docker compose -f docker-compose.dev.yml --profile testing up -d cucumber-report
# Visit: http://localhost:5050
```

**See also:** [BDD Tests README](../integration/tests/acceptance/README.md) for detailed documentation.

### E2E Tests (End-to-End)

Browser-based tests using Playwright.

**Location:** `webserver/web/`

**Technology:** Playwright + TypeScript

**Browsers:** Chromium, Firefox, WebKit (Safari)

**What they test:**
- UI interactions
- Component rendering
- User workflows
- WebSocket connections
- Video processing
- Filter configuration
- Multi-source management
- Camera functionality

**Test Suites:**
- Drawer functionality
- Filter configuration
- Filter toggle behavior
- Multi-source management
- Panel synchronization
- Resolution control
- Source removal
- UI validation
- WebSocket management
- Image selection

#### Prerequisites

**Test video required:**
```bash
# Generate test video if missing
./scripts/tools/generate-video.sh

# Extract frames if missing
./scripts/tools/extract-frames.sh
```

**Services running:**
```bash
./scripts/dev/start.sh
```

#### Run E2E Tests

**Via script (recommended):**
```bash
./scripts/test/e2e.sh
```

**Options:**
- `--chromium, -c` - Run only in Chromium
- `--firefox, -f` - Run only in Firefox
- `--webkit, -w` - Run only in WebKit/Safari
- `--headed` - Run in headed mode (visible browser)
- `--debug` - Debug mode
- `--ui` - UI mode
- `--workers=N` - Number of parallel workers (default: 25)
- `--dev` - Development environment (default)
- `--staging` - Staging environment
- `--prod` - Production environment
- `--help, -h` - Show help

**Examples:**
```bash
# Fast: Chromium only
./scripts/test/e2e.sh --chromium

# All browsers
./scripts/test/e2e.sh

# Visible browser for debugging
./scripts/test/e2e.sh --headed --chromium

# Production environment
./scripts/test/e2e.sh --prod --chromium

# Staging environment
./scripts/test/e2e.sh --staging --chromium
```

**With Docker:**
```bash
./scripts/test/integration.sh e2e
```

**View reports:**
```bash
npx playwright show-report
```

**View HTML report:**
```bash
docker compose -f docker-compose.dev.yml --profile testing up -d e2e-report-viewer
# Visit: http://localhost:5051
```

## Running All Tests

### Quick Validation

**Unit tests only:**
```bash
./scripts/test/unit-tests.sh
```

**Integration tests (BDD + E2E):**
```bash
./scripts/test/integration.sh all
```

### Full Test Suite

**Run all test types sequentially:**
```bash
# 1. Unit tests
./scripts/test/unit-tests.sh

# 2. BDD tests
./scripts/test/integration.sh backend

# 3. E2E tests
./scripts/test/e2e.sh --chromium
```

## Git Hooks

Automated validation runs before commits and pushes.

### Pre-commit Hook

**What it runs:**
- Unit tests (Go, Frontend, C++)
- Linters (ESLint, golangci-lint, clang-tidy)
- Frontend build validation

**Install:**
```bash
./scripts/hooks/install.sh
```

**Skip when needed:**
```bash
git commit --no-verify
```

**See:** [Git Hooks Documentation](../README.md#git-hooks)

### Pre-push Hook

**What it runs:**
- Full validation with all browsers
- E2E tests across Chromium, Firefox, WebKit

**Skip when needed:**
```bash
git push --no-verify
```

## Code Quality

### Linters

**Run all linters:**
```bash
./scripts/test/linters.sh
```

**Auto-fix issues:**
```bash
./scripts/test/linters.sh --fix
```

**Linters used:**
- **Frontend:** ESLint + Prettier
- **Golang:** golangci-lint
- **C++:** clang-tidy + clang-format

**Individual linters:**
```bash
# Frontend
cd webserver/web
npm run lint
npm run lint:fix

# Golang
golangci-lint run ./...

# C++
clang-tidy -p . cpp_accelerator/**/*.cpp
```

**See:** [README Testing Section](../README.md#testing-code-quality)

### Coverage Reports

Coverage reports are generated when running tests with coverage enabled.

**Generate coverage:**
```bash
./scripts/test/coverage.sh
```

**Options:**
- `--skip-frontend` - Skip frontend coverage
- `--skip-golang` - Skip Golang coverage
- `--skip-cpp` - Skip C++ coverage
- `--help` - Show usage

**View reports:**
```bash
# Start coverage report viewer
docker-compose -f docker-compose.dev.yml --profile coverage up coverage-report-viewer

# Visit: http://localhost:5052
```

**Report locations:**
- Frontend: `coverage/frontend/index.html`
- Golang: `coverage/golang/index.html`
- C++: `coverage/cpp/html/index.html`

## Test Organization

### Directory Structure

```
integration/tests/acceptance/
├── features/              # Gherkin feature files
├── steps/                # Step definitions
├── testdata/             # Test data and checksums
└── scripts/              # Setup and utilities

webserver/web/
├── src/                  # Source code
└── tests/                # E2E and unit tests

webserver/pkg/
└── **/*_test.go         # Go unit tests

cpp_accelerator/
└── **/*_test.cpp        # C++ unit tests (when implemented)
```

## Troubleshooting

### Services Not Running

**Error:** `Flipt is not accessible at http://localhost:8081`

**Solution:**
```bash
./scripts/dev/start.sh
```

### Certificate Errors

E2E tests use `InsecureSkipVerify: true` for development. If you see certificate errors:
- Check TLS configuration
- Ensure services are running on correct ports
- Verify environment variables

### Test Video Missing

**Error:** `Test video not found: data/test-data/videos/e2e-test.mp4`

**Solution:**
```bash
./scripts/tools/generate-video.sh
```

### Proto Files Not Generated

**Error:** Missing proto types in tests

**Solution:**
```bash
./integration/tests/acceptance/scripts/setup.sh
```

## Additional Resources

- [BDD Tests README](../integration/tests/acceptance/README.md) - Detailed BDD documentation
- [README Testing Section](../README.md#testing-code-quality) - Quick reference
- [Git Hooks](../README.md#git-hooks) - Pre-commit and pre-push hooks

