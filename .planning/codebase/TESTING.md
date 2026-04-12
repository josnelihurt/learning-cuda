# Testing Patterns

**Analysis Date:** 2026-04-12

## Test Framework

**Go:**
- **Runner:** Standard `go test` with testify assertions
- **Assertion Library:** `github.com/stretchr/testify` (assert, require, mock)
- **Config:** No separate config file; uses Go's built-in test flags
- **Scripts:** `./scripts/test/unit-tests.sh --skip-frontend` for Go-only tests

**TypeScript:**
- **Runner:** Vitest 1.2.0
- **Config:** `webserver/web/vitest.config.ts`
- **Assertion Library:** Vitest built-in (`expect`, `vi`)
- **Test Environment:** happy-dom for DOM testing
- **Scripts:** `npm run test` (watch), `npm run test:coverage`, `./scripts/test/unit-tests.sh --skip-golang`

**C++:**
- **Runner:** GoogleTest (gtest)
- **Config:** Bazel build targets (e.g., `//cpp_accelerator/core:logger_test`)
- **Scripts:** `bazel test //cpp_accelerator/...`, `./scripts/test/unit-tests.sh --skip-golang --skip-frontend`

**Run Commands:**
```bash
# All tests
./scripts/test/unit-tests.sh

# Go tests
go test -race ./webserver/pkg/...

# Frontend tests
cd webserver/web && npm run test
cd webserver/web && npm run test:coverage

# C++ tests
bazel test //cpp_accelerator/...

# BDD acceptance tests (requires services running)
go test ./integration/tests/acceptance -run TestFeatures -v
```

## Test File Organization

**Location:**
- **Go:** Co-located with source (`upload_image_use_case_test.go` next to `upload_image_use_case.go`)
- **TypeScript:** Co-located with source (`websocket-frame-transport.test.ts` next to `websocket-frame-transport.ts`)
- **C++:** Co-located with source (`logger_test.cpp` next to `logger.cc`)

**Naming:**
- **Go:** `<source>_test.go`
- **TypeScript:** `<source>.test.ts`
- **C++:** `<source>_test.cpp` or `<source>_test.cc`

**Structure:**
```
webserver/pkg/
├── application/
│   ├── upload_image_use_case.go
│   └── upload_image_use_case_test.go
└── infrastructure/
    ├── filesystem/
    │   ├── static_image_repository.go
    │   └── static_image_repository_test.go

webserver/web/src/
├── infrastructure/
│   └── transport/
│       ├── websocket-frame-transport.ts
│       └── websocket-frame-transport.test.ts
└── domain/
    └── value-objects/
        ├── FilterData.ts
        └── FilterData.test.ts

cpp_accelerator/
├── core/
│   ├── logger.cc
│   └── logger_test.cpp
└── infrastructure/
    └── cpu/
        ├── grayscale_filter.cc
        └── grayscale_filter_test.cpp
```

## Test Structure

**Go Suite Organization:**
```go
func TestNewUploadImageUseCase(t *testing.T) {
    // Arrange
    repo := new(mockStaticImageRepository)

    // Act
    sut := NewUploadImageUseCase(repo)

    // Assert
    require.NotNil(t, sut)
    assert.Equal(t, repo, sut.repository)
}
```

**Go Patterns:**
- Use AAA comments (Arrange/Act/Assert)
- Use `sut` for system under test
- Table-driven tests for multiple cases
- Test data builders: `makeValidPNGData()`, `makeValidImage()`
- Naming: `Success_`, `Error_`, `Edge_` prefix in table test names
- Mocks: Embed `mock.Mock`, use `On().Return()`, `AssertExpectations()`

**TypeScript Suite Organization:**
```typescript
describe('WebSocketService', () => {
  it('sends frames when websocket is connected', () => {
    const mockWs = { readyState: 1, send: vi.fn() };
    const service = makeService();
    (service as any).ws = mockWs;

    service.sendFrame('data:image/png;base64,test', 32, 32, makeFilters('grayscale'), 'gpu');

    expect(mockWs.send).toHaveBeenCalled();
  });
});
```

**TypeScript Patterns:**
- Use AAA comments in complex tests
- Use `sut` for system under test
- Helper functions for test data: `makeFilters()`, `makeStats()`, `makeService()`
- Nested `describe` blocks for related functionality
- Mocking: `vi.fn()`, `vi.mock()`, `vi.spyOn()`

**C++ Suite Organization:**
```cpp
class GrayscaleFilterTest : public ::testing::Test {
protected:
  void SetUp() override {
    image_loader_ = std::make_unique<ImageLoader>();
    ASSERT_TRUE(image_loader_->is_valid());
  }

  void TearDown() override {}

  std::unique_ptr<ImageLoader> image_loader_;
};

TEST_F(GrayscaleFilterTest, AppliesBT601GrayscaleSuccessfully) {
  GrayscaleFilter filter(GrayscaleAlgorithm::BT601);
  std::vector<unsigned char> output(image_loader_->width() * image_loader_->height() * 1);
  FilterContext context = CreateGrayscaleFilterContext(...);

  bool result = filter.Apply(context);

  ASSERT_TRUE(result);
  EXPECT_EQ(context.output.channels, 1);
}
```

**C++ Patterns:**
- Test fixtures with `SetUp()` / `TearDown()`
- Helper functions for test data: `CreateGrayscaleFilterContext()`
- Use `TEST_F` for fixture tests, `TEST` for simple tests
- Naming: `AppliesBT601GrayscaleSuccessfully`, `HandlesSingleChannelInput`, `FailsWithInvalidInput`
- Member variables with trailing underscore: `image_loader_`

**Patterns:**
- **Setup:** Use table-driven tests (Go), `beforeEach` (TypeScript), test fixtures (C++)
- **Teardown:** `defer` (Go), `afterEach` (TypeScript), `TearDown()` (C++)
- **Assertion:** Use `require` for fatal failures (Go), `assert` for non-fatal

## Mocking

**Go Framework:** testify/mock

**Patterns:**
```go
type mockStaticImageRepository struct {
    mock.Mock
}

func (m *mockStaticImageRepository) Save(ctx context.Context, filename string, data []byte) (*domain.StaticImage, error) {
    args := m.Called(ctx, filename, data)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*domain.StaticImage), args.Error(1)
}

// In test
mockRepo.On("Save", mock.Anything, tt.filename, tt.fileData).
    Return(tt.mockImage, tt.mockError).
    Once()

mockRepo.AssertExpectations(t)
```

**TypeScript Framework:** Vitest built-in (`vi`)

**Patterns:**
```typescript
const mockLogger = {
  info: vi.fn(),
  debug: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
  initialize: vi.fn(),
  shutdown: vi.fn(),
} as any;

// Spy on existing methods
vi.spyOn(document, 'querySelector').mockImplementation((selector: string) => {
  if (selector === 'toast-container') return mockToast;
  return null;
});

// Restore mocks
vi.restoreAllMocks();
```

**C++ Framework:** No explicit mocking framework; use fakes/stubs

**Patterns:**
- Create test doubles implementing interfaces
- Use test fixtures with setup/teardown
- GoogleMock available but not extensively used in current codebase

**What to Mock:**
- External dependencies (databases, APIs, file system)
- Infrastructure concerns (HTTP clients, gRPC clients)
- Time for deterministic tests
- Random number generators

**What NOT to Mock:**
- Domain value objects (e.g., `FilterData`, `ImageData`)
- Simple data structures
- Pure functions (test them directly instead)

## Fixtures and Factories

**Test Data:**
```go
// Go builders
func makeValidPNGData() []byte {
    return []byte{137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13}
}

func makeValidImage() *domain.StaticImage {
    return &domain.StaticImage{
        ID:          "test-upload",
        DisplayName: "Test Upload",
        Path:        "/data/static_images/test-upload.png",
        IsDefault:   false,
    }
}
```

```typescript
// TypeScript helpers
const makeFilters = (...ids: string[]) => ids.map((id) => new FilterData(id));
const makeStats = () => ({ updateWebSocketStatus: vi.fn() });
const makeService = () => new WebSocketService(makeStats(), makeCamera(), makeToast());
```

**Location:**
- Go: Inline in test files or shared test package
- TypeScript: Inline in test files
- C++: Helper functions in test file or anonymous namespace

## Coverage

**Requirements:**
- **Go:** No explicit target enforced; aim for >80% on critical paths
- **TypeScript:** 80% threshold enforced in `vitest.config.ts`:
  - lines: 80%
  - functions: 80%
  - branches: 80%
  - statements: 80%
- **C++:** No explicit target; use equivalence tests for CUDA/CPU comparison

**View Coverage:**
```bash
# Frontend
cd webserver/web && npm run test:coverage
# Reports in: coverage/frontend/

# All coverage
./scripts/test/coverage.sh
```

**Coverage Configuration:**
- TypeScript reports: JSON, HTML, lcov, text
- Output directory: `coverage/frontend/`
- Excludes: node_modules, static, tests, proto, *.d.ts

## Test Types

**Unit Tests:**
- **Go:** Test individual functions/methods in isolation
- **TypeScript:** Test components and services in isolation
- **C++:** Test filter algorithms, data structures
- Use mocks for external dependencies
- Fast execution (milliseconds)

**Integration Tests:**
- **Go:** Located in `integration/tests/`
- **TypeScript:** Not explicitly separated; use unit tests for integration
- **C++:** Equivalence tests verify CPU/CUDA produce identical results
- May require real dependencies (database, file system)

**BDD Acceptance Tests:**
- **Framework:** godog (Cucumber for Go)
- **Location:** `integration/tests/acceptance/`
- **Scripts:** `go test ./integration/tests/acceptance -run TestFeatures -v`
- Requires services running
- Feature files with Gherkin syntax

**E2E Tests:**
- **Framework:** Playwright
- **Config:** `webserver/web/playwright.config.ts`
- **Scripts:** `./scripts/test/e2e.sh --chromium`
- Tests user workflows in browser
- Requires full application running

## Common Patterns

**Async Testing (TypeScript):**
```typescript
it('should initialize services', async () => {
  await expect(element.initialize()).resolves.not.toThrow();
  expect(mockLogger.info).toHaveBeenCalledWith('Initializing app-root...');
});
```

**Error Testing (Go):**
```go
{
    name: "Error_FileTooLarge",
    filename: "large.png",
    fileData: makeLargePNGData(),
    assertResult: func(t *testing.T, image *domain.StaticImage, err error) {
        assert.ErrorIs(t, err, errFileTooLarge)
        assert.Nil(t, image)
    },
}
```

**Table-Driven Tests (Go):**
```go
tests := []struct {
    name         string
    input        string
    expected     string
    assertResult func(t *testing.T, result string)
}{
    {name: "Success_ValidInput", input: "test", expected: "test", assertResult: ...},
    {name: "Error_InvalidInput", input: "", expected: "", assertResult: ...},
}

for _, tt := range tests {
    t.Run(tt.name, func(t *testing.T) {
        result := sut.Process(tt.input)
        tt.assertResult(t, result)
    })
}
```

**Component Testing (TypeScript/Lit):**
```typescript
it('should render control sections', async () => {
  await element.updateComplete;
  const controlSections = element.shadowRoot!.querySelectorAll('.control-section');
  expect(controlSections.length).toBeGreaterThan(0);
});
```

---

*Testing analysis: 2026-04-12*
