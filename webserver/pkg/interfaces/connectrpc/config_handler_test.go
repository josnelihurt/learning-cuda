package connectrpc

import (
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Tests
func TestNewConfigHandler(t *testing.T) {
	// Arrange
	mockConfigManager := &config.Manager{}

	// Act
	sut := NewConfigHandler(
		nil, // getStreamConfigUseCase
		nil, // syncFlagsUseCase
		nil, // listInputsUseCase
		nil, // evaluateFFUseCase
		nil, // getSystemInfoUseCase
		mockConfigManager,
	)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, mockConfigManager, sut.configManager)
}
