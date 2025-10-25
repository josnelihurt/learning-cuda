package build

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewBuildInfo(t *testing.T) {
	tests := []struct {
		name     string
		envVars  map[string]string
		expected *Info
	}{
		{
			name:    "default values",
			envVars: map[string]string{},
			expected: &Info{
				Version:    "1.0.0",
				Branch:     "main",
				BuildTime:  "unknown",
				CommitHash: "unknown",
			},
		},
		{
			name: "environment variables override",
			envVars: map[string]string{
				"APP_VERSION": "2.0.0",
				"APP_BRANCH":  "develop",
				"BUILD_TIME":  "2024-10-25T12:00:00Z",
				"COMMIT_HASH": "abc123",
			},
			expected: &Info{
				Version:    "2.0.0",
				Branch:     "develop",
				BuildTime:  "2024-10-25T12:00:00Z",
				CommitHash: "abc123",
			},
		},
		{
			name: "partial environment variables",
			envVars: map[string]string{
				"APP_VERSION": "3.0.0",
				"COMMIT_HASH": "def456",
			},
			expected: &Info{
				Version:    "3.0.0",
				Branch:     "main",
				BuildTime:  "unknown",
				CommitHash: "def456",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: Set environment variables
			for key, value := range tt.envVars {
				os.Setenv(key, value)
			}
			defer func() {
				// Cleanup
				for key := range tt.envVars {
					os.Unsetenv(key)
				}
			}()

			// Act
			result := NewBuildInfo()

			// Assert
			assert.Equal(t, tt.expected.Version, result.Version)
			assert.Equal(t, tt.expected.Branch, result.Branch)
			assert.Equal(t, tt.expected.BuildTime, result.BuildTime)
			assert.Equal(t, tt.expected.CommitHash, result.CommitHash)
		})
	}
}
