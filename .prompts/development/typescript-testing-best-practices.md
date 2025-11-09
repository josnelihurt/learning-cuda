# TypeScript Testing Patterns - AI Reference

Apply these patterns when creating/refactoring TypeScript/Vitest tests. All test code in English.

## Rules

ALWAYS:
- Use AAA comments (Arrange/Act/Assert)
- Use `vi.fn()` and `vi.mock()` for mocking
- Use `sut` for system under test
- Use table-driven tests for multiple cases
- Use `assertResult` function per test case
- Use Uber naming: `Success_`, `Error_`, `Edge_`
- Use typed errors with custom Error classes
- Use test data builders `makeXXX()`
- Use `describe` and `it` with descriptive names
- Use `expect` from Vitest

NEVER:
- Manual mocks with complex setup
- If/else in assertions
- Redundant inline comments
- Verbose test names with underscores
- Mixing test concerns in single test

## Assertions

```typescript
expect(result).toBeDefined()        // Check not undefined
expect(result).toBeNull()           // Check null
expect(result).toBe(expected)       // Strict equality
expect(result).toEqual(expected)    // Deep equality
expect(result).toThrow('message')   // Error throwing
expect(mockFn).toHaveBeenCalled()   // Mock verification
expect(mockFn).toHaveBeenCalledWith(args) // Mock with args
```

## Naming

Test names: `{Category}_{Description}` in CamelCase

- `Success_ValidInput`, `Success_CPUProcessing`
- `Error_InvalidInput`, `Error_Timeout`
- `Edge_EmptyList`, `Edge_NullPointer`

Variables:
- `sut` - System under test
- `mockXXX` - Mocks
- `makeXXX()` - Builders
- `errXXX` - Typed errors

---

## Complete Template

Apply this pattern to all tests:

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ClassName } from './ClassName';

// Mock
const mockDependency = vi.fn();

// Test data builders
const makeValidData = () => ({
  id: 1,
  name: 'Test Data',
});

const makeInvalidData = () => ({
  id: 0,
  name: '',
});

// Simple test
describe('ClassName', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Success Cases', () => {
    it('Success_ValidInput', () => {
      // Arrange
      const sut = new ClassName();
      const input = makeValidData();

      // Act
      const result = sut.method(input);

      // Assert
      expect(result).toBeDefined();
      expect(result.id).toBe(1);
    });

    it('Success_WithMock', () => {
      // Arrange
      const mockService = { process: vi.fn().mockReturnValue('result') };
      const sut = new ClassName(mockService);
      const input = makeValidData();

      // Act
      const result = sut.method(input);

      // Assert
      expect(result).toBe('result');
      expect(mockService.process).toHaveBeenCalledWith(input);
    });
  });

  describe('Error Cases', () => {
    it('Error_InvalidInput', () => {
      // Arrange
      const sut = new ClassName();
      const input = makeInvalidData();

      // Act / Assert
      expect(() => sut.method(input)).toThrow('Invalid input');
    });

    it('Error_ServiceFailure', () => {
      // Arrange
      const mockService = { 
        process: vi.fn().mockImplementation(() => {
          throw new Error('Service failed');
        })
      };
      const sut = new ClassName(mockService);
      const input = makeValidData();

      // Act / Assert
      expect(() => sut.method(input)).toThrow('Service failed');
    });
  });

  describe('Edge Cases', () => {
    it('Edge_EmptyInput', () => {
      // Arrange
      const sut = new ClassName();
      const input = {};

      // Act
      const result = sut.method(input);

      // Assert
      expect(result).toBeDefined();
    });

    it('Edge_NullInput', () => {
      // Arrange
      const sut = new ClassName();

      // Act / Assert
      expect(() => sut.method(null)).toThrow('Input cannot be null');
    });
  });
});

// Table-driven test
describe('ClassName - Table Driven', () => {
  const testCases = [
    {
      name: 'Success_ValidData',
      input: makeValidData(),
      expected: 'success',
      assertResult: (result: string) => {
        expect(result).toBe('success');
      }
    },
    {
      name: 'Error_InvalidData',
      input: makeInvalidData(),
      expected: null,
      assertResult: (result: string | null) => {
        expect(result).toBeNull();
      }
    }
  ];

  testCases.forEach(({ name, input, assertResult }) => {
    it(name, () => {
      // Arrange
      const sut = new ClassName();

      // Act
      const result = sut.method(input);

      // Assert
      assertResult(result);
    });
  });
});
```

---

## Quick Reference

**Mock setup:**
```typescript
const mockFn = vi.fn();
mockFn.mockReturnValue('result');
mockFn.mockImplementation(() => 'custom result');
mockFn.mockRejectedValue(new Error('error'));
```

**Builders:**
```typescript
const makeValidUser = () => ({
  id: 1,
  name: 'John Doe',
  email: 'john@example.com'
});
```

**Functional assertion:**
```typescript
assertResult: (result: User) => {
  expect(result).toBeDefined();
  expect(result.name).toBe('John Doe');
}
```

**Typed errors:**
```typescript
class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

// In test:
expect(() => sut.validate('')).toThrow(ValidationError);
```

**Async testing:**
```typescript
it('Success_AsyncOperation', async () => {
  // Arrange
  const sut = new ClassName();
  const input = makeValidData();

  // Act
  const result = await sut.asyncMethod(input);

  // Assert
  expect(result).toBeDefined();
});
```

---

## Checklist

When creating tests:

1. [ ] AAA comments present
2. [ ] `vi.fn()` and `vi.mock()` used (not manual mocks)
3. [ ] `sut` variable naming
4. [ ] Table-driven for multiple cases
5. [ ] `assertResult` per test case (no if/else)
6. [ ] Uber naming applied (Success_, Error_, Edge_)
7. [ ] Typed errors with custom Error classes
8. [ ] Test data builders `makeXXX()`
9. [ ] `vi.clearAllMocks()` in beforeEach
10. [ ] No redundant comments
11. [ ] Descriptive test names
12. [ ] Proper async/await handling
