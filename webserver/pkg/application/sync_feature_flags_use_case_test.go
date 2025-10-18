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

func makeValidFeatureFlags() []domain.FeatureFlag {
	return []domain.FeatureFlag{
		{
			Key:          "flag1",
			Name:         "Test Flag 1",
			Type:         domain.BooleanFlagType,
			Enabled:      true,
			DefaultValue: true,
		},
		{
			Key:          "flag2",
			Name:         "Test Flag 2",
			Type:         domain.VariantFlagType,
			Enabled:      true,
			DefaultValue: "variant1",
		},
	}
}

func makeEmptyFeatureFlags() []domain.FeatureFlag {
	return []domain.FeatureFlag{}
}

func TestNewSyncFeatureFlagsUseCase(t *testing.T) {
	// Arrange
	repo := new(mockFeatureFlagRepository)

	// Act
	sut := NewSyncFeatureFlagsUseCase(repo)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, repo, sut.repository)
}

func TestSyncFeatureFlagsUseCase_Execute(t *testing.T) {
	var (
		errSyncFailed = errors.New("sync failed")
	)

	tests := []struct {
		name         string
		flags        []domain.FeatureFlag
		mockError    error
		assertResult func(t *testing.T, err error)
	}{
		{
			name:      "Success_ValidFlags",
			flags:     makeValidFeatureFlags(),
			mockError: nil,
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:      "Success_EmptyFlags",
			flags:     makeEmptyFeatureFlags(),
			mockError: nil,
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:      "Error_SyncFailed",
			flags:     makeValidFeatureFlags(),
			mockError: errSyncFailed,
			assertResult: func(t *testing.T, err error) {
				assert.ErrorIs(t, err, errSyncFailed)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockRepo := new(mockFeatureFlagRepository)
			mockRepo.On("SyncFlags",
				mock.Anything,
				tt.flags,
			).Return(tt.mockError).Once()

			sut := NewSyncFeatureFlagsUseCase(mockRepo)
			ctx := context.Background()

			// Act
			err := sut.Execute(ctx, tt.flags)

			// Assert
			tt.assertResult(t, err)
			mockRepo.AssertExpectations(t)
		})
	}
}

func TestSyncFeatureFlagsUseCase_ContextCancellation(t *testing.T) {
	// Arrange
	mockRepo := new(mockFeatureFlagRepository)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mockRepo.On("SyncFlags",
		mock.Anything,
		mock.Anything,
	).Return(ctx.Err()).Once()

	sut := NewSyncFeatureFlagsUseCase(mockRepo)

	// Act
	err := sut.Execute(ctx, makeValidFeatureFlags())

	// Assert
	assert.Error(t, err)
	mockRepo.AssertExpectations(t)
}