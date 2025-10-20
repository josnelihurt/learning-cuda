package application

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

func TestUploadVideoUseCase_Execute(t *testing.T) {
	tests := []struct {
		name        string
		filename    string
		fileData    []byte
		setup       func(*MockVideoRepository, string, string)
		expectError error
	}{
		{
			name:     "successfully uploads valid MP4",
			filename: "test.mp4",
			fileData: []byte("fake video data"),
			setup: func(repo *MockVideoRepository, videosDir, previewsDir string) {
				repo.On("Save", mock.Anything, mock.AnythingOfType("*domain.Video")).Return(nil)
			},
			expectError: nil,
		},
		{
			name:     "returns error for invalid format",
			filename: "test.avi",
			fileData: []byte("fake video"),
			setup: func(repo *MockVideoRepository, videosDir, previewsDir string) {
			},
			expectError: ErrInvalidFormat,
		},
		{
			name:     "returns error for file too large",
			filename: "test.mp4",
			fileData: make([]byte, 101*1024*1024),
			setup: func(repo *MockVideoRepository, videosDir, previewsDir string) {
			},
			expectError: ErrFileTooLarge,
		},
		{
			name:     "returns error when repository save fails",
			filename: "test.mp4",
			fileData: []byte("fake video"),
			setup: func(repo *MockVideoRepository, videosDir, previewsDir string) {
				repo.On("Save", mock.Anything, mock.AnythingOfType("*domain.Video")).
					Return(errors.New("save failed"))
			},
			expectError: errors.New("save failed"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := new(MockVideoRepository)
			videosDir := t.TempDir()
			previewsDir := t.TempDir()
			tt.setup(repo, videosDir, previewsDir)
			sut := NewUploadVideoUseCase(repo, videosDir, previewsDir)

			result, err := sut.Execute(context.Background(), tt.fileData, tt.filename)

			if tt.expectError != nil {
				assert.Error(t, err)
				assert.Nil(t, result)
				if errors.Is(tt.expectError, ErrInvalidFormat) {
					assert.ErrorIs(t, err, ErrInvalidFormat)
				} else if errors.Is(tt.expectError, ErrFileTooLarge) {
					assert.ErrorIs(t, err, ErrFileTooLarge)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
			}
		})
	}
}

func TestUploadVideoUseCase_validateFormat(t *testing.T) {
	tests := []struct {
		name        string
		filename    string
		expectError bool
	}{
		{
			name:        "accepts MP4 files",
			filename:    "video.mp4",
			expectError: false,
		},
		{
			name:        "rejects AVI files",
			filename:    "video.avi",
			expectError: true,
		},
		{
			name:        "rejects MKV files",
			filename:    "video.mkv",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sut := &UploadVideoUseCase{}

			err := sut.validateFormat(tt.filename)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUploadVideoUseCase_validateSize(t *testing.T) {
	tests := []struct {
		name        string
		fileData    []byte
		expectError bool
	}{
		{
			name:        "accepts files under 100MB",
			fileData:    make([]byte, 50*1024*1024),
			expectError: false,
		},
		{
			name:        "rejects files over 100MB",
			fileData:    make([]byte, 101*1024*1024),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sut := &UploadVideoUseCase{}

			err := sut.validateSize(tt.fileData)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
