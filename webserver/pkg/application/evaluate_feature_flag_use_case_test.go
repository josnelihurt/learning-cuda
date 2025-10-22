package application

import (
	"context"
	"errors"
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type mockFeatureFlagRepository struct {
	mock.Mock
}

func (m *mockFeatureFlagRepository) EvaluateBoolean(ctx context.Context, flagKey, entityID string) (*domain.FeatureFlagEvaluation, error) {
	args := m.Called(ctx, flagKey, entityID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.FeatureFlagEvaluation), args.Error(1)
}

func (m *mockFeatureFlagRepository) EvaluateVariant(ctx context.Context, flagKey, entityID string) (*domain.FeatureFlagEvaluation, error) {
	args := m.Called(ctx, flagKey, entityID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.FeatureFlagEvaluation), args.Error(1)
}

func (m *mockFeatureFlagRepository) SyncFlags(ctx context.Context, flags []domain.FeatureFlag) error {
	args := m.Called(ctx, flags)
	return args.Error(0)
}

func (m *mockFeatureFlagRepository) GetFlag(ctx context.Context, flagKey string) (*domain.FeatureFlag, error) {
	args := m.Called(ctx, flagKey)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.FeatureFlag), args.Error(1)
}

func makeBooleanEvaluation(result, success bool) *domain.FeatureFlagEvaluation {
	return &domain.FeatureFlagEvaluation{
		FlagKey:      "test_flag",
		EntityID:     "test_entity",
		Result:       result,
		Success:      success,
		UsedFallback: !success,
	}
}

func makeVariantEvaluation(result string, success bool) *domain.FeatureFlagEvaluation {
	return &domain.FeatureFlagEvaluation{
		FlagKey:      "test_flag",
		EntityID:     "test_entity",
		Result:       result,
		Success:      success,
		UsedFallback: !success,
	}
}

func TestNewEvaluateFeatureFlagUseCase(t *testing.T) {
	// Arrange
	repo := new(mockFeatureFlagRepository)

	// Act
	sut := NewEvaluateFeatureFlagUseCase(repo)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, repo, sut.repository)
}

func TestEvaluateFeatureFlagUseCase_EvaluateBoolean(t *testing.T) {
	var (
		errRepositoryFailed = errors.New("repository failed")
	)

	tests := []struct {
		name          string
		flagKey       string
		entityID      string
		fallbackValue bool
		mockResult    *domain.FeatureFlagEvaluation
		mockError     error
		assertResult  func(t *testing.T, result bool, err error)
	}{
		{
			name:          "Success_BooleanTrue",
			flagKey:       "feature_enabled",
			entityID:      "user_123",
			fallbackValue: false,
			mockResult:    makeBooleanEvaluation(true, true),
			mockError:     nil,
			assertResult: func(t *testing.T, result bool, err error) {
				assert.NoError(t, err)
				assert.True(t, result)
			},
		},
		{
			name:          "Success_BooleanFalse",
			flagKey:       "feature_disabled",
			entityID:      "user_456",
			fallbackValue: true,
			mockResult:    makeBooleanEvaluation(false, true),
			mockError:     nil,
			assertResult: func(t *testing.T, result bool, err error) {
				assert.NoError(t, err)
				assert.False(t, result)
			},
		},
		{
			name:          "Error_BooleanWithFallback",
			flagKey:       "broken_flag",
			entityID:      "user_789",
			fallbackValue: true,
			mockResult:    nil,
			mockError:     errRepositoryFailed,
			assertResult: func(t *testing.T, result bool, err error) {
				assert.NoError(t, err)
				assert.True(t, result)
			},
		},
		{
			name:          "Edge_BooleanEvaluationFailedUseFallback",
			flagKey:       "failed_flag",
			entityID:      "user_000",
			fallbackValue: false,
			mockResult:    makeBooleanEvaluation(true, false),
			mockError:     nil,
			assertResult: func(t *testing.T, result bool, err error) {
				assert.NoError(t, err)
				assert.False(t, result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockRepo := new(mockFeatureFlagRepository)
			mockRepo.On("EvaluateBoolean",
				mock.Anything,
				tt.flagKey,
				tt.entityID,
			).Return(tt.mockResult, tt.mockError).Once()

			sut := NewEvaluateFeatureFlagUseCase(mockRepo)
			ctx := context.Background()

			// Act
			result, err := sut.EvaluateBoolean(ctx, tt.flagKey, tt.entityID, tt.fallbackValue)

			// Assert
			tt.assertResult(t, result, err)
			mockRepo.AssertExpectations(t)
		})
	}
}

func TestEvaluateFeatureFlagUseCase_EvaluateVariant(t *testing.T) {
	var (
		errRepositoryFailed = errors.New("repository failed")
	)

	tests := []struct {
		name          string
		flagKey       string
		entityID      string
		fallbackValue string
		mockResult    *domain.FeatureFlagEvaluation
		mockError     error
		assertResult  func(t *testing.T, result string, err error)
	}{
		{
			name:          "Success_VariantResult",
			flagKey:       "theme_variant",
			entityID:      "user_123",
			fallbackValue: "default",
			mockResult:    makeVariantEvaluation("dark", true),
			mockError:     nil,
			assertResult: func(t *testing.T, result string, err error) {
				assert.NoError(t, err)
				assert.Equal(t, "dark", result)
			},
		},
		{
			name:          "Error_VariantWithFallback",
			flagKey:       "broken_variant",
			entityID:      "user_456",
			fallbackValue: "fallback_value",
			mockResult:    nil,
			mockError:     errRepositoryFailed,
			assertResult: func(t *testing.T, result string, err error) {
				assert.NoError(t, err)
				assert.Equal(t, "fallback_value", result)
			},
		},
		{
			name:          "Edge_VariantEvaluationFailedUseFallback",
			flagKey:       "failed_variant",
			entityID:      "user_789",
			fallbackValue: "safe_default",
			mockResult:    makeVariantEvaluation("wrong", false),
			mockError:     nil,
			assertResult: func(t *testing.T, result string, err error) {
				assert.NoError(t, err)
				assert.Equal(t, "safe_default", result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockRepo := new(mockFeatureFlagRepository)
			mockRepo.On("EvaluateVariant",
				mock.Anything,
				tt.flagKey,
				tt.entityID,
			).Return(tt.mockResult, tt.mockError).Once()

			sut := NewEvaluateFeatureFlagUseCase(mockRepo)
			ctx := context.Background()

			// Act
			result, err := sut.EvaluateVariant(ctx, tt.flagKey, tt.entityID, tt.fallbackValue)

			// Assert
			tt.assertResult(t, result, err)
			mockRepo.AssertExpectations(t)
		})
	}
}
