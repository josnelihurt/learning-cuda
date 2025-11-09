# Go Testing Patterns - AI Reference

Apply these patterns when creating/refactoring Go tests. All test code in English.

## Rules

ALWAYS:
- Use AAA comments (Arrange/Act/Assert)
- Use testify/mock (embed `mock.Mock`)
- Use `sut` for system under test
- Use table-driven tests for multiple cases
- Use `assertResult` function per test case
- Use Uber naming: `Success_`, `Error_`, `Edge_`
- Use typed errors with `errors.Is`
- Use test data builders `makeXXX()`

NEVER:
- Manual mocks with `processFunc`
- If/else in assertions
- Redundant inline comments
- Verbose test names with underscores

## Assertions

```go
require.NoError(t, err)    // Stops test if fails (critical)
require.NotNil(t, result)  // Like GoogleTest ASSERT_*

assert.Equal(t, x, y)      // Continues if fails
assert.Greater(t, x, 10)   // Like GoogleTest EXPECT_*
```

## Naming

Test names: `{Category}_{Description}` in CamelCase

- `Success_ValidInput`, `Success_CPUProcessing`
- `Error_InvalidInput`, `Error_Timeout`
- `Edge_EmptyList`, `Edge_NilPointer`

Variables:
- `sut` - System under test
- `mockXXX` - Mocks
- `makeXXX()` - Builders
- `errXXX` - Typed errors

---

## Complete Template

Apply this pattern to all tests:

```go
package application

import (
    "context"
    "errors"
    "testing"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "github.com/stretchr/testify/require"
)

// Mock
type mockRepository struct {
    mock.Mock
}

func (m *mockRepository) GetByID(ctx context.Context, id int) (*Entity, error) {
    args := m.Called(ctx, id)
    if args.Get(0) == nil {
        return nil, args.Error(1)
    }
    return args.Get(0).(*Entity), args.Error(1)
}

func (m *mockRepository) Save(ctx context.Context, entity *Entity) error {
    args := m.Called(ctx, entity)
    return args.Error(0)
}

// Test data builders
func makeValidEntity() *Entity {
    return &Entity{
        ID:   1,
        Name: "Test Entity",
    }
}

func makeInvalidEntity() *Entity {
    return &Entity{ID: 0, Name: ""}
}

// Simple test
func TestNewService(t *testing.T) {
    // Arrange
    repo := new(mockRepository)

    // Act
    sut := NewService(repo)

    // Assert
    require.NotNil(t, sut)
    assert.Equal(t, repo, sut.repo)
}

// Table-driven test
func TestService_Execute(t *testing.T) {
    var (
        errInvalidEntity = errors.New("invalid entity")
        errSaveFailed    = errors.New("save failed")
    )

    tests := []struct {
        name         string
        input        *Entity
        mockResult   *Entity
        mockError    error
        assertResult func(t *testing.T, result *Entity, err error)
    }{
        {
            name:       "Success_ValidEntity",
            input:      makeValidEntity(),
            mockResult: makeValidEntity(),
            mockError:  nil,
            assertResult: func(t *testing.T, result *Entity, err error) {
                assert.NoError(t, err)
                require.NotNil(t, result)
                assert.Equal(t, 1, result.ID)
            },
        },
        {
            name:       "Error_InvalidEntity",
            input:      makeInvalidEntity(),
            mockResult: nil,
            mockError:  errInvalidEntity,
            assertResult: func(t *testing.T, result *Entity, err error) {
                assert.ErrorIs(t, err, errInvalidEntity)
                assert.Nil(t, result)
            },
        },
        {
            name:       "Error_SaveFailed",
            input:      makeValidEntity(),
            mockResult: nil,
            mockError:  errSaveFailed,
            assertResult: func(t *testing.T, result *Entity, err error) {
                assert.ErrorIs(t, err, errSaveFailed)
                assert.Nil(t, result)
            },
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // Arrange
            mockRepo := new(mockRepository)
            mockRepo.On("Save",
                mock.Anything,
                tt.input,
            ).Return(tt.mockError).Once()

            sut := NewService(mockRepo)
            ctx := context.Background()

            // Act
            result, err := sut.Execute(ctx, tt.input)

            // Assert
            tt.assertResult(t, result, err)
            mockRepo.AssertExpectations(t)
        })
    }
}

// Context cancellation test
func TestService_ContextCancellation(t *testing.T) {
    // Arrange
    mockRepo := new(mockRepository)
    ctx, cancel := context.WithCancel(context.Background())
    cancel()

    mockRepo.On("Save",
        mock.Anything,
        mock.Anything,
    ).Return(ctx.Err()).Once()

    sut := NewService(mockRepo)

    // Act
    result, err := sut.Execute(ctx, makeValidEntity())

    // Assert
    assert.Error(t, err)
    assert.Nil(t, result)
    mockRepo.AssertExpectations(t)
}
```

---

## Quick Reference

**Mock setup:**
```go
mockDep := new(mockDependency)
mockDep.On("MethodName", 
    mock.Anything,
    exactParam,
).Return(result, err).Once()
```

**Builders:**
```go
func makeValidData() *Data {
    return &Data{Field: "value"}
}
```

**Functional assertion:**
```go
assertResult: func(t *testing.T, result *Data, err error) {
    assert.NoError(t, err)
    require.NotNil(t, result)
    assert.Equal(t, expected, result.Field)
}
```

**Typed errors:**
```go
var errSpecific = errors.New("specific error")

assertResult: func(t *testing.T, result *Data, err error) {
    assert.ErrorIs(t, err, errSpecific)
    assert.Nil(t, result)
}
```

---

## Checklist

When creating tests:

1. [ ] AAA comments present
2. [ ] testify/mock used (not manual mocks)
3. [ ] `sut` variable naming
4. [ ] Table-driven for multiple cases
5. [ ] `assertResult` per test case (no if/else)
6. [ ] Uber naming applied (Success_, Error_, Edge_)
7. [ ] Typed errors with `errors.Is`
8. [ ] Test data builders `makeXXX()`
9. [ ] `AssertExpectations(t)` called
10. [ ] No redundant comments
