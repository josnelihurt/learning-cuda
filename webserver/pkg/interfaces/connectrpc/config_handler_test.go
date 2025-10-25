package connectrpc

import (
	"sync"
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Tests
func TestNewConfigHandler(t *testing.T) {
	// Arrange
	var currentLoader *loader.Loader
	loaderMutex := &sync.RWMutex{}
	mockConfigManager := &config.Manager{}

	// Act
	sut := NewConfigHandler(
		nil, // getStreamConfigUseCase
		nil, // syncFlagsUseCase
		nil, // listInputsUseCase
		nil, // evaluateFFUseCase
		nil, // getSystemInfoUseCase
		nil, // registry
		&currentLoader,
		loaderMutex,
		mockConfigManager,
	)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, &currentLoader, sut.currentLoader)
	assert.Equal(t, loaderMutex, sut.loaderMutex)
	assert.Equal(t, mockConfigManager, sut.configManager)
}
