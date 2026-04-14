package featureflags

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGoffRepository_ValidateConfig_ErrorWhenFileDoesNotExist(t *testing.T) {
	repo := NewGoffRepository(filepath.Join(t.TempDir(), "missing.goff.yaml"))

	err := repo.ValidateConfig()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "goff file not found")
}

func TestGoffRepository_ValidateConfig_ErrorWhenYamlIsInvalid(t *testing.T) {
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "flags.goff.yaml")
	require.NoError(t, os.WriteFile(filePath, []byte("invalid: ["), 0o644))
	repo := NewGoffRepository(filePath)

	err := repo.ValidateConfig()

	require.Error(t, err)
	assert.Contains(t, err.Error(), "unmarshal goff file")
}
