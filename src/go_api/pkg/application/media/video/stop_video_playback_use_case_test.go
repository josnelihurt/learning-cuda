package video

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStopVideoPlaybackUseCase_Execute_MissingSessionID(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	uc := NewStopVideoPlaybackUseCase(manager)

	input := StopVideoPlaybackUseCaseInput{}

	// Act
	_, err := uc.Execute(context.Background(), input)

	// Assert
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrVideoPlaybackMissingSession)
}

func TestStopVideoPlaybackUseCase_Execute_SessionNotFound(t *testing.T) {
	// Arrange
	manager := NewVideoSessionManager()
	uc := NewStopVideoPlaybackUseCase(manager)

	input := StopVideoPlaybackUseCaseInput{
		SessionID: "non-existent",
	}

	// Act
	_, err := uc.Execute(context.Background(), input)

	// Assert
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrVideoPlaybackNotRunning)
}
