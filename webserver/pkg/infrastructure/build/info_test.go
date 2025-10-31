package build

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewBuildInfo(t *testing.T) {
	result := NewBuildInfo()

	assert.NotNil(t, result)
	assert.Equal(t, "1.0.0", result.Version)
	assert.Equal(t, "main", result.Branch)
	assert.Equal(t, "unknown", result.BuildTime)
	assert.Equal(t, "unknown", result.CommitHash)
}
