package video

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFileVideoRepository_List(t *testing.T) {
	tests := []struct {
		name          string
		setup         func(t *testing.T) (string, string)
		expectedCount int
		expectedError bool
	}{
		{
			name: "lists videos from directory",
			setup: func(t *testing.T) (string, string) {
				videosDir := t.TempDir()
				previewsDir := t.TempDir()

				err := os.WriteFile(filepath.Join(videosDir, "test.mp4"), []byte("fake video"), 0600)
				require.NoError(t, err)

				return videosDir, previewsDir
			},
			expectedCount: 1,
			expectedError: false,
		},
		{
			name: "returns empty list when directory is empty",
			setup: func(t *testing.T) (string, string) {
				return t.TempDir(), t.TempDir()
			},
			expectedCount: 0,
			expectedError: false,
		},
		{
			name: "ignores non-mp4 files",
			setup: func(t *testing.T) (string, string) {
				videosDir := t.TempDir()
				previewsDir := t.TempDir()

				err := os.WriteFile(filepath.Join(videosDir, "test.avi"), []byte("avi"), 0600)
				require.NoError(t, err)
				err = os.WriteFile(filepath.Join(videosDir, "valid.mp4"), []byte("mp4"), 0600)
				require.NoError(t, err)

				return videosDir, previewsDir
			},
			expectedCount: 1,
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			videosDir, previewsDir := tt.setup(t)
			sut := NewFileVideoRepository(videosDir, previewsDir)

			result, err := sut.List(context.Background())

			if tt.expectedError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Len(t, result, tt.expectedCount)
			}
		})
	}
}

func TestFileVideoRepository_GetByID(t *testing.T) {
	tests := []struct {
		name          string
		videoID       string
		setup         func(t *testing.T) (string, string)
		expectError   bool
		expectedVideo string
	}{
		{
			name:    "returns video when found",
			videoID: "test",
			setup: func(t *testing.T) (string, string) {
				videosDir := t.TempDir()
				previewsDir := t.TempDir()

				err := os.WriteFile(filepath.Join(videosDir, "test.mp4"), []byte("video"), 0600)
				require.NoError(t, err)

				return videosDir, previewsDir
			},
			expectError:   false,
			expectedVideo: "test",
		},
		{
			name:    "returns error when not found",
			videoID: "nonexistent",
			setup: func(t *testing.T) (string, string) {
				return t.TempDir(), t.TempDir()
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			videosDir, previewsDir := tt.setup(t)
			sut := NewFileVideoRepository(videosDir, previewsDir)

			result, err := sut.GetByID(context.Background(), tt.videoID)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.Equal(t, tt.expectedVideo, result.ID)
			}
		})
	}
}
