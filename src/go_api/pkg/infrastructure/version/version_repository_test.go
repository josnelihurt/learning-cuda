package version

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewVersionRepository(t *testing.T) {
	repo := NewVersionRepository()
	require.NotNil(t, repo)

	impl, ok := repo.(*RepositoryImpl)
	require.True(t, ok)
	assert.NotEmpty(t, impl.goVersionPath)
	assert.NotEmpty(t, impl.cppVersionPath)
	assert.NotEmpty(t, impl.protoVersionPath)
}

func TestRepositoryImpl_GetGoVersion(t *testing.T) {
	repo := NewVersionRepository()

	version := repo.GetGoVersion()

	if _, err := os.Stat(filepath.Join("..", "..", "..", "..", "webserver", "VERSION")); err == nil {
		assert.NotEqual(t, UnknownValue, version)
		assert.NotEmpty(t, version)
	} else {
		assert.Equal(t, UnknownValue, version)
	}
}

func TestRepositoryImpl_GetCppVersion(t *testing.T) {
	repo := NewVersionRepository()

	version := repo.GetCppVersion()

	if _, err := os.Stat(filepath.Join("..", "..", "..", "..", "cpp_accelerator", "VERSION")); err == nil {
		assert.NotEqual(t, UnknownValue, version)
		assert.NotEmpty(t, version)
	} else {
		assert.Equal(t, UnknownValue, version)
	}
}

func TestRepositoryImpl_GetProtoVersion(t *testing.T) {
	repo := NewVersionRepository()

	version := repo.GetProtoVersion()

	if _, err := os.Stat(filepath.Join("..", "..", "..", "..", "proto", "VERSION")); err == nil {
		assert.NotEqual(t, UnknownValue, version)
		assert.NotEmpty(t, version)
	} else {
		assert.Equal(t, UnknownValue, version)
	}
}

func TestRepositoryImpl_readVersionFile_NonExistent(t *testing.T) {
	repo := &RepositoryImpl{
		goVersionPath: "/nonexistent/path/VERSION",
	}

	version := repo.readVersionFile("/nonexistent/path/VERSION")
	assert.Equal(t, UnknownValue, version)
}

func TestFindProjectRoot(t *testing.T) {
	root := findProjectRoot()
	assert.NotEmpty(t, root)

	webserverVersionPath := filepath.Join(root, "webserver", "VERSION")
	_, err := os.Stat(webserverVersionPath)
	assert.NoError(t, err, "webserver/VERSION should exist in project root")
}
