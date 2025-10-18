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

func makeBooleanFlag(key string, enabled bool) domain.FeatureFlag {
	return domain.FeatureFlag{
		Key:          key,
		Name:         "Test Flag",
		Type:         domain.BooleanFlagType,
		Enabled:      enabled,
		DefaultValue: false,
	}
}

func makeVariantFlag(key string, enabled bool) domain.FeatureFlag {
	return domain.FeatureFlag{
		Key:          key,
		Name:         "Variant Flag",
		Type:         domain.VariantFlagType,
		Enabled:      enabled,
		DefaultValue: "default",
	}
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
			name: "Success_SyncMultipleFlags",
			flags: []domain.FeatureFlag{
				makeBooleanFlag("flag_1", true),
				makeBooleanFlag("flag_2", false),
				makeVariantFlag("flag_3", true),
			},
			mockError: nil,
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name:      "Success_SyncEmptyFlags",
			flags:     []domain.FeatureFlag{},
			mockError: nil,
			assertResult: func(t *testing.T, err error) {
				assert.NoError(t, err)
			},
		},
		{
			name: "Error_SyncFailed",
			flags: []domain.FeatureFlag{
				makeBooleanFlag("flag_1", true),
			},
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
