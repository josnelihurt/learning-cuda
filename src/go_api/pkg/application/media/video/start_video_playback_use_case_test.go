package video

import (
	"context"
	"testing"
	"time"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

type MockStreamVideoPlayer struct {
	mock.Mock
}

func (m *MockStreamVideoPlayer) Play(
	ctx context.Context,
	frameCallback func(*domain.Image, int, time.Duration) error,
) error {
	args := m.Called(ctx, frameCallback)
	return args.Error(0)
}

func TestStartVideoPlaybackUseCase_Execute_MissingVideoID(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	repo := &MockVideoRepository{}

	uc := NewStartVideoPlaybackUseCase(context.Background(), manager, repo,
		func(videoPath string) (StreamVideoPlayer, error) {
			return &MockStreamVideoPlayer{}, nil
		},
		func(browserSessionID string) (StreamVideoPeer, error) {
			return &MockPeer{}, nil
		},
	)

	input := StartVideoPlaybackUseCaseInput{
		SessionID: "test-session",
	}

	// Act
	_, err := uc.Execute(context.Background(), input)

	// Assert
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrVideoPlaybackMissingVideoID)
}

func TestStartVideoPlaybackUseCase_Execute_MissingSessionID(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	repo := &MockVideoRepository{}

	uc := NewStartVideoPlaybackUseCase(context.Background(), manager, repo,
		func(videoPath string) (StreamVideoPlayer, error) {
			return &MockStreamVideoPlayer{}, nil
		},
		func(browserSessionID string) (StreamVideoPeer, error) {
			return &MockPeer{}, nil
		},
	)

	input := StartVideoPlaybackUseCaseInput{
		VideoID: "test-video",
	}

	// Act
	_, err := uc.Execute(context.Background(), input)

	// Assert
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrVideoPlaybackMissingSession)
}

