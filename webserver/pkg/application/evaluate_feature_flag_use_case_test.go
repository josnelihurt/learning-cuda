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

func makeValidBooleanEvaluation() *domain.FeatureFlagEvaluation {
	return &domain.FeatureFlagEvaluation{
		FlagKey:      "test_flag",
		EntityID:     "test_entity",
		Result:       true,
		Success:      true,
		UsedFallback: false,
	}
}

func makeValidVariantEvaluation() *domain.FeatureFlagEvaluation {
	return &domain.FeatureFlagEvaluation{
		FlagKey:      "test_flag",
		EntityID:     "test_entity",
		Result:       "variant_value",
		Success:      true,
		UsedFallback: false,
	}
}

func makeFailedEvaluation() *domain.FeatureFlagEvaluation {
	return &domain.FeatureFlagEvaluation{
		FlagKey:      "test_flag",
		EntityID:     "test_entity",
		Result:       nil,
		Success:      false,
		UsedFallback: true,
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
		errRepository = errors.New("repository error")
	)

	tests := []struct {
		name         string
		flagKey      string
		entityID     string
		fallbackValue bool
		mockResult   *domain.FeatureFlagEvaluation
		mockError    error
		assertResult func(t *testing.T, result bool, err error)
	}{
		{
			name:         "Success_ValidEvaluation",
			flagKey:      "test_flag",
			entityID:     "test_entity",
			fallbackValue: false,
			mockResult:   makeValidBooleanEvaluation(),
			mockError:    nil,
			assertResult: func(t *testing.T, result bool, err error) {
				assert.NoError(t, err)
				assert.True(t, result)
			},
		},
		{
			name:         "Success_RepositoryErrorUsesFallback",
			flagKey:      "test_flag",
			entityID:     "test_entity",
			fallbackValue: true,
			mockResult:   nil,
			mockError:    errRepository,
			assertResult: func(t *testing.T, result bool, err error) {
				assert.NoError(t, err)
				assert.True(t, result)
			},
		},
		{
			name:         "Success_EvaluationFailedUsesFallback",
			flagKey:      "test_flag",
			entityID:     "test_entity",
			fallbackValue: false,
			mockResult:   makeFailedEvaluation(),
			mockError:    nil,
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
		errRepository = errors.New("repository error")
	)

	tests := []struct {
		name         string
		flagKey      string
		entityID     string
		fallbackValue string
		mockResult   *domain.FeatureFlagEvaluation
		mockError    error
		assertResult func(t *testing.T, result string, err error)
	}{
		{
			name:         "Success_ValidEvaluation",
			flagKey:      "test_flag",
			entityID:     "test_entity",
			fallbackValue: "default_variant",
			mockResult:   makeValidVariantEvaluation(),
			mockError:    nil,
			assertResult: func(t *testing.T, result string, err error) {
				assert.NoError(t, err)
				assert.Equal(t, "variant_value", result)
			},
		},
		{
			name:         "Success_RepositoryErrorUsesFallback",
			flagKey:      "test_flag",
			entityID:     "test_entity",
			fallbackValue: "fallback_variant",
			mockResult:   nil,
			mockError:    errRepository,
			assertResult: func(t *testing.T, result string, err error) {
				assert.NoError(t, err)
				assert.Equal(t, "fallback_variant", result)
			},
		},
		{
			name:         "Success_EvaluationFailedUsesFallback",
			flagKey:      "test_flag",
			entityID:     "test_entity",
			fallbackValue: "default_variant",
			mockResult:   makeFailedEvaluation(),
			mockError:    nil,
			assertResult: func(t *testing.T, result string, err error) {
				assert.NoError(t, err)
				assert.Equal(t, "default_variant", result)
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

func TestEvaluateFeatureFlagUseCase_ContextCancellation(t *testing.T) {
	// Arrange
	mockRepo := new(mockFeatureFlagRepository)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mockRepo.On("EvaluateBoolean",
		mock.Anything,
		mock.Anything,
		mock.Anything,
	).Return(nil, ctx.Err()).Once()

	sut := NewEvaluateFeatureFlagUseCase(mockRepo)

	// Act
	result, err := sut.EvaluateBoolean(ctx, "test_flag", "test_entity", true)

	// Assert
	assert.NoError(t, err)
	assert.True(t, result)
	mockRepo.AssertExpectations(t)
}