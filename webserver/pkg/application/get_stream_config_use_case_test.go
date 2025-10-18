package application

import (
	"context"
	"errors"
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func makeValidStreamConfig() config.StreamConfig {
	return config.StreamConfig{
		TransportFormat:   "json",
		WebsocketEndpoint: "/ws",
	}
}

func makeAlternativeStreamConfig() config.StreamConfig {
	return config.StreamConfig{
		TransportFormat:   "protobuf",
		WebsocketEndpoint: "/ws",
	}
}

func TestNewGetStreamConfigUseCase(t *testing.T) {
	// Arrange
	mockRepo := new(mockFeatureFlagRepository)
	evaluateUC := NewEvaluateFeatureFlagUseCase(mockRepo)
	defaultCfg := makeValidStreamConfig()

	// Act
	sut := NewGetStreamConfigUseCase(evaluateUC, defaultCfg)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, evaluateUC, sut.evaluateFFUseCase)
	assert.Equal(t, defaultCfg, sut.defaultConfig)
}

func TestGetStreamConfigUseCase_Execute(t *testing.T) {
	var (
		errEvaluationFailed = errors.New("evaluation failed")
	)

	tests := []struct {
		name           string
		defaultConfig  config.StreamConfig
		mockResult     string
		mockError      error
		assertResult   func(t *testing.T, result *config.StreamConfig, err error)
	}{
		{
			name:          "Success_DefaultTransportFormat",
			defaultConfig: makeValidStreamConfig(),
			mockResult:    "json",
			mockError:     nil,
			assertResult: func(t *testing.T, result *config.StreamConfig, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "json", result.TransportFormat)
				assert.Equal(t, "/ws", result.WebsocketEndpoint)
			},
		},
		{
			name:          "Success_AlternativeTransportFormat",
			defaultConfig: makeValidStreamConfig(),
			mockResult:    "protobuf",
			mockError:     nil,
			assertResult: func(t *testing.T, result *config.StreamConfig, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "protobuf", result.TransportFormat)
				assert.Equal(t, "/ws", result.WebsocketEndpoint)
			},
		},
		{
			name:          "Success_EvaluationFailedUsesFallback",
			defaultConfig: makeValidStreamConfig(),
			mockResult:    "",
			mockError:     errEvaluationFailed,
			assertResult: func(t *testing.T, result *config.StreamConfig, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "json", result.TransportFormat)
				assert.Equal(t, "/ws", result.WebsocketEndpoint)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockRepo := new(mockFeatureFlagRepository)
			mockRepo.On("EvaluateVariant",
				mock.Anything,
				"ws_transport_format",
				"stream_transport_format",
			).Return(&domain.FeatureFlagEvaluation{
				FlagKey:      "ws_transport_format",
				EntityID:     "stream_transport_format",
				Result:       tt.mockResult,
				Success:      tt.mockError == nil,
				UsedFallback: false,
			}, tt.mockError).Once()

			evaluateUC := NewEvaluateFeatureFlagUseCase(mockRepo)
			sut := NewGetStreamConfigUseCase(evaluateUC, tt.defaultConfig)
			ctx := context.Background()

			// Act
			result, err := sut.Execute(ctx)

			// Assert
			tt.assertResult(t, result, err)
			mockRepo.AssertExpectations(t)
		})
	}
}

func TestGetStreamConfigUseCase_ContextCancellation(t *testing.T) {
	// Arrange
	mockRepo := new(mockFeatureFlagRepository)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mockRepo.On("EvaluateVariant",
		mock.Anything,
		"ws_transport_format",
		"stream_transport_format",
	).Return(nil, ctx.Err()).Once()

	evaluateUC := NewEvaluateFeatureFlagUseCase(mockRepo)
	sut := NewGetStreamConfigUseCase(evaluateUC, makeValidStreamConfig())

	// Act
	result, err := sut.Execute(ctx)

	// Assert
	assert.NoError(t, err)
	require.NotNil(t, result)
	assert.Equal(t, "json", result.TransportFormat)
	assert.Equal(t, "/ws", result.WebsocketEndpoint)
	mockRepo.AssertExpectations(t)
}