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

func makeDefaultStreamConfig() config.StreamConfig {
	return config.StreamConfig{
		TransportFormat:   "json",
		WebsocketEndpoint: "/ws",
	}
}

func TestNewGetStreamConfigUseCase(t *testing.T) {
	// Arrange
	repo := new(mockFeatureFlagRepository)
	evaluateUC := NewEvaluateFeatureFlagUseCase(repo)
	defaultCfg := makeDefaultStreamConfig()

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
		mockEvaluation *domain.FeatureFlagEvaluation
		mockError      error
		assertResult   func(t *testing.T, result *config.StreamConfig, err error)
	}{
		{
			name:           "Success_DefaultTransportFormat",
			defaultConfig:  makeDefaultStreamConfig(),
			mockEvaluation: makeVariantEvaluation("json", true),
			mockError:      nil,
			assertResult: func(t *testing.T, result *config.StreamConfig, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "json", result.TransportFormat)
				assert.Equal(t, "/ws", result.WebsocketEndpoint)
			},
		},
		{
			name: "Success_CustomTransportFormat",
			defaultConfig: config.StreamConfig{
				TransportFormat:   "json",
				WebsocketEndpoint: "/custom",
			},
			mockEvaluation: makeVariantEvaluation("msgpack", true),
			mockError:      nil,
			assertResult: func(t *testing.T, result *config.StreamConfig, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, "msgpack", result.TransportFormat)
				assert.Equal(t, "/custom", result.WebsocketEndpoint)
			},
		},
		{
			name:           "Success_FallbackOnRepositoryError",
			defaultConfig:  makeDefaultStreamConfig(),
			mockEvaluation: nil,
			mockError:      errEvaluationFailed,
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
			).Return(tt.mockEvaluation, tt.mockError).Once()

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
