# Coding Conventions

**Analysis Date:** 2026-04-12

## Naming Patterns

**Files:**
- **Go:** `snake_case.go` (e.g., `upload_image_use_case.go`, `static_image_repository.go`)
- **TypeScript:** `kebab-case.ts` (e.g., `websocket-frame-transport.ts`, `stats-panel.ts`)
- **C++:** `snake_case.cc` / `snake_case.h` (e.g., `grayscale_filter.cc`, `logger.h`)
- **Test files:** Append `_test.go` (Go), `.test.ts` (TypeScript), `_test.cpp` (C++)

**Functions:**
- **Go:** `PascalCase` for exported, `camelCase` for unexported (e.g., `NewUploadImageUseCase()`, `isPNGFormat()`)
- **TypeScript:** `camelCase` (e.g., `sendFrame()`, `getConnectionStatus()`)
- **C++:** `PascalCase` for public methods (e.g., `Apply()`, `GetType()`)

**Variables:**
- **Go:** `camelCase` (e.g., `maxFileSize`, `errFileTooLarge`)
- **TypeScript:** `camelCase` (e.g., `imageData`, `kernelSize`)
- **C++:** `snake_case` for members, `camelCase` for locals (e.g., `image_loader_`, `filter_bt601`)

**Types:**
- **Go:** `PascalCase` (e.g., `StaticImage`, `UploadImageUseCase`)
- **TypeScript:** `PascalCase` (e.g., `FilterData`, `WebSocketService`, `IFileService`)
- **C++:** `PascalCase` (e.g., `FilterContext`, `GrayscaleAlgorithm`)

**Constants:**
- **Go:** `camelCase` for package-level (e.g., `maxFileSize`), `UPPER_SNAKE_CASE` for special cases
- **TypeScript:** `PascalCase` for enums/objects (e.g., `ConnectionStatus`), `camelCase` for const vars
- **C++:** `kPascalCase` or `UPPER_SNAKE_CASE` (e.g., `kEnvironment`, `MAX_SIZE`)

**Interfaces:**
- **Go:** Simple `PascalCase` names (e.g., `StaticImageRepository`, `VideoPlayer`)
- **TypeScript:** `IPascalCase` prefix (e.g., `IFileService`, `IWebSocketService`, `IFrameTransportService`)

## Code Style

**Formatting:**
- **Go:** `gofmt` and `goimports` (enforced by golangci-lint)
- **TypeScript:** Prettier with config at `webserver/web/.prettierrc`:
  - Single quotes, semicolons, trailing commas (ES5)
  - 100 char line width, 2 space tabs
  - LF line endings
- **C++:** No auto-formatter detected; manual formatting with clang-format recommended

**Linting:**
- **Go:** golangci-lint with extensive rules (`.golangci.yml`):
  - Core: errcheck, gosimple, govet, staticcheck, unused
  - Style: gofmt, goimports, misspell, revive, stylecheck
  - Bugs: gosec, nilerr, contextcheck
  - Complexity: gocyclo (max 15), dupl (threshold 150)
  - Disabled linters for tests: errcheck, goconst, gosec
- **TypeScript:** ESLint with config at `webserver/web/.eslintrc.json`:
  - Extends: eslint:recommended, @typescript-eslint/recommended, plugin:lit/recommended
  - Rules: warn on `any` and unused vars (with `_` prefix ignore), error on prefer-const/no-var
  - Ignores: node_modules, static, tests, proto, *.config.ts

## Import Organization

**Order (Go):**
1. Standard library
2. Third-party packages
3. Internal packages (`github.com/jrb/cuda-learning/...`)

**Order (TypeScript):**
1. External dependencies (including generated protobuf imports)
2. Internal imports with `@/` alias or relative paths
3. Type-only imports grouped separately

**Path Aliases:**
- **TypeScript:** `@` maps to `./src` (configured in `vitest.config.ts` and `vite.config.ts`)

**Examples:**
```typescript
// External imports first
import { LitElement, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';

// Internal imports second
import { logger } from '../observability/otel-logger';
import type { IWebSocketService } from '../../domain/interfaces/IWebSocketService';
```

## Error Handling

**Go Patterns:**
- Use package-level `var` for error constants: `errFileTooLarge = errors.New("file too large")`
- Wrap errors with context: `fmt.Errorf("failed to save image: %w", err)`
- Return errors as last return value: `func (...) (*Type, error)`
- Check errors immediately: `if err != nil { return nil, err }`
- Use `assert.ErrorIs` and `assert.NoError` in tests
- Never ignore errors (enforced by errcheck linter, except in tests)

**TypeScript Patterns:**
- Throw `Error` objects with descriptive messages: `throw new Error('Filter type cannot be empty')`
- Use try-catch for async operations
- Return rejected promises for async errors
- Validate inputs early in constructors/methods
- Use type guards for runtime type checking

**C++ Patterns:**
- Return `bool` for success/failure in filter operations
- Return `Result<T, E>` type for operations that can fail (see `cpp_accelerator/core/result.h`)
- Use `EXPECT_FALSE` / `ASSERT_TRUE` in tests for error conditions

**Error Wrapping (Go):**
```go
// Good: wrap with context
return nil, fmt.Errorf("failed to save image: %w", err)

// Good: define package-level errors
var (
    errFileTooLarge  = errors.New("file too large")
    errInvalidFormat = errors.New("invalid format")
)
```

## Logging

**Go Framework:** zerolog (structured logging)

**Patterns:**
```go
// Get logger from context
log := logger.FromContext(ctx)

// Structured logging with fields
log.Info().
    Str("filename", filename).
    Int("file_size", len(fileData)).
    Msg("Image uploaded successfully")

// Error logging
log.Error().Err(err).Str("collector_url", url).Msg("Failed to forward logs")

// Global logger (use sparingly)
logger.Global().Debug().Msg("Starting server")
```

**TypeScript Framework:** Custom OTEL logger at `webserver/web/src/infrastructure/observability/otel-logger.ts`

**Patterns:**
```typescript
// Structured logging with key-value pairs
logger.info('WebSocket connected');
logger.error('Error parsing WebSocket message', {
  'error.message': error instanceof Error ? error.message : String(error),
});
logger.debug('Frame sent via WebSocket', {
  width: image.getWidth(),
  height: image.getHeight(),
  filterCount: filters.length,
});
```

**OpenTelemetry Integration:**
- Go: Use `otel.Tracer()` for spans, `span.SetAttributes()`, `span.RecordError()`
- TypeScript: Use `telemetryService.withSpanAsync()` for tracing async operations

## Comments

**When to Comment:**
- Exported functions/methods MUST have godoc comments (Go)
- Complex algorithms or non-obvious logic
- TODO/FIXME comments for technical debt (tracked by godox linter)
- Configuration explanations
- Workarounds for bugs in dependencies

**JSDoc/TSDoc:**
- Use JSDoc for public APIs in TypeScript
- Not extensively used in current codebase
- Prefer descriptive function/parameter names over comments

**Example (Go godoc):**
```go
// NewStaticImageRepository creates a repository for managing static images
// in the specified directory path.
func NewStaticImageRepository(directory string) *StaticImageRepository {
    return &StaticImageRepository{
        directory: directory,
    }
}
```

## Function Design

**Size:**
- Keep functions focused on single responsibility
- Go: Functions under ~50 lines preferred (gocyclo linter flags complexity > 15)
- Extract complex logic into helper functions

**Parameters:**
- **Go:** Context first: `func (uc *UseCase) Execute(ctx context.Context, ...)`
- **Go:** Prefer few parameters; group related params in structs
- **TypeScript:** Use parameter objects for complex functions: `sendFrame(base64Data: string, width: number, height: number, ...)`
- **TypeScript:** Use options pattern for optional parameters
- **C++:** Pass by const reference for large objects: `const FilterContext& context`

**Return Values:**
- **Go:** Multiple returns for errors: `func (...) (*Type, error)`
- **Go:** Return zero values on error (nil for pointers, empty for slices)
- **TypeScript:** Return types explicitly declared; use union types for error cases
- **C++:** Return Status/result objects or bool for success

## Module Design

**Exports:**
- **Go:** Exported identifiers start with `PascalCase`, unexported with `camelCase`
- **TypeScript:** Use `export class` / `export function` / `export interface`
- **TypeScript:** Default exports avoided; prefer named exports

**Barrel Files:**
- **TypeScript:** `application/di/index.ts` provides dependency injection container
- **TypeScript:** Avoid deep relative imports; use `@/` alias instead

**Package Structure (Go):**
```
webserver/pkg/
├── application/     # Use cases (upload_image_use_case.go)
├── domain/          # Entities and interfaces (image_repository.go)
├── infrastructure/  # External concerns (filesystem/static_image_repository.go)
└── interfaces/      # HTTP/WebSocket handlers (connectrpc/file_handler.go)
```

**Directory Structure (TypeScript):**
```
webserver/web/src/
├── application/     # Services and DI container
├── domain/          # Value objects and interfaces
├── infrastructure/  # Transport, logging, external services
└── components/      # Lit Web Components
```

**Clean Architecture Principles:**
- Dependencies point inward (domain → application → infrastructure)
- Use cases orchestrate domain logic
- Infrastructure implements domain interfaces
- No domain code depends on infrastructure details

---

*Convention analysis: 2026-04-12*
