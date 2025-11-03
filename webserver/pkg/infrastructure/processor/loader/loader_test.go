package loader

import (
	"testing"

	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Tests
func TestCurrentAPIVersion(t *testing.T) {
	// Test that the API version constant is defined
	assert.Equal(t, "2.1.0", CurrentAPIVersion)
}

func TestIsCompatible(t *testing.T) {
	tests := []struct {
		name     string
		v1       string
		v2       string
		expected bool
	}{
		{
			name:     "Success_SameMajorVersion",
			v1:       "2.0.0",
			v2:       "2.1.0",
			expected: true,
		},
		{
			name:     "Error_DifferentMajorVersion",
			v1:       "1.0.0",
			v2:       "2.0.0",
			expected: false,
		},
		{
			name:     "Success_ExactMatch",
			v1:       "2.0.0",
			v2:       "2.0.0",
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			compatible := isCompatible(tt.v1, tt.v2)

			// Assert
			assert.Equal(t, tt.expected, compatible)
		})
	}
}

func TestGetMajorVersion(t *testing.T) {
	tests := []struct {
		name     string
		version  string
		expected int
	}{
		{
			name:     "Success_ValidVersion",
			version:  "2.1.0",
			expected: 2,
		},
		{
			name:     "Success_SingleNumber",
			version:  "3",
			expected: 3,
		},
		{
			name:     "Error_EmptyVersion",
			version:  "",
			expected: 0,
		},
		{
			name:     "Error_InvalidVersion",
			version:  "invalid",
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Act
			major := getMajorVersion(tt.version)

			// Assert
			assert.Equal(t, tt.expected, major)
		})
	}
}

func TestLoader_GetVersion(t *testing.T) {
	// Arrange
	expectedVersion := "2.0.0"
	loader := &Loader{
		apiVersion: expectedVersion,
	}

	// Act
	version := loader.GetVersion()

	// Assert
	assert.Equal(t, expectedVersion, version)
}

func TestLoader_IsCompatibleWith(t *testing.T) {
	tests := []struct {
		name          string
		apiVersion    string
		loaderVersion string
		expected      bool
	}{
		{
			name:          "Success_CompatibleVersions",
			apiVersion:    "2.0.0",
			loaderVersion: "2.1.0",
			expected:      true,
		},
		{
			name:          "Error_IncompatibleVersions",
			apiVersion:    "1.0.0",
			loaderVersion: "2.0.0",
			expected:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			loader := &Loader{
				apiVersion: tt.loaderVersion,
			}

			// Act
			compatible := loader.IsCompatibleWith(tt.apiVersion)

			// Assert
			assert.Equal(t, tt.expected, compatible)
		})
	}
}

func TestLoader_CachedCapabilities(t *testing.T) {
	// Arrange
	expectedCaps := &pb.LibraryCapabilities{
		ApiVersion:        "2.0.0",
		LibraryVersion:    "1.0.0",
		SupportsStreaming: true,
		BuildDate:         "2024-01-01",
		BuildCommit:       "abc123",
	}
	loader := &Loader{
		capabilities: expectedCaps,
	}

	// Act
	caps := loader.CachedCapabilities()

	// Assert
	require.NotNil(t, caps)
	assert.Equal(t, expectedCaps, caps)
}

// TODO: CGO Testing Limitations
// Problem: CGO functions (dlopen, dlsym, callInitFn, etc.) cannot be easily mocked in unit tests
// because they are C functions called through Go's CGO interface. Direct mocking is not possible.
//
// Possible solutions:
// 1. Integration tests with real shared libraries (.so files)
// 2. Build test doubles as separate shared libraries for testing
// 3. Use build tags to compile different versions for testing
// 4. Create wrapper interfaces that can be mocked, but this requires refactoring the loader
// 5. Use dependency injection to inject CGO function pointers (complex)
//
// Current approach: Test only pure Go functions and simple getters/setters
// For full testing, consider integration tests with actual CUDA libraries
