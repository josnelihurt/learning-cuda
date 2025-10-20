package video

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewGoVideoPlayer(t *testing.T) {
	tests := []struct {
		name        string
		setup       func(t *testing.T) string
		expectError bool
	}{
		{
			name: "creates player for existing video",
			setup: func(t *testing.T) string {
				tmpDir := t.TempDir()
				videoPath := filepath.Join(tmpDir, "test.mp4")
				err := os.WriteFile(videoPath, []byte("fake video"), 0600)
				require.NoError(t, err)
				return videoPath
			},
			expectError: false,
		},
		{
			name: "returns error for non-existent video",
			setup: func(t *testing.T) string {
				return "/nonexistent/video.mp4"
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			videoPath := tt.setup(t)

			sut, err := NewGoVideoPlayer(videoPath)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, sut)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, sut)
			}
		})
	}
}

func TestGoVideoPlayer_Stop(t *testing.T) {
	tmpDir := t.TempDir()
	videoPath := filepath.Join(tmpDir, "test.mp4")
	err := os.WriteFile(videoPath, []byte("fake video"), 0600)
	require.NoError(t, err)

	sut, err := NewGoVideoPlayer(videoPath)
	require.NoError(t, err)

	err = sut.Stop(context.Background())

	assert.NoError(t, err)
	assert.True(t, sut.stopped)
}

func TestGoVideoPlayer_Play(t *testing.T) {
	tmpDir := t.TempDir()
	videoPath := filepath.Join(tmpDir, "test.mp4")
	err := os.WriteFile(videoPath, []byte("fake video"), 0600)
	require.NoError(t, err)

	sut, err := NewGoVideoPlayer(videoPath)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	callback := func(frameData []byte, frameNumber int, timestamp time.Duration) error {
		return nil
	}

	err = sut.Play(ctx, videoPath, callback)

	assert.Error(t, err)
	assert.Equal(t, context.DeadlineExceeded, err)
}
