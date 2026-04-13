package application

import (
	"context"
	"testing"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func TestNewListInputsUseCase(t *testing.T) {
	// Arrange
	mockRepo := new(MockVideoRepository)

	// Act
	sut := NewListInputsUseCase(mockRepo)

	// Assert
	require.NotNil(t, sut)
}

func TestListInputsUseCase_Execute(t *testing.T) {
	tests := []struct {
		name         string
		assertResult func(t *testing.T, result []InputSource, err error)
	}{
		{
			name: "Success_ReturnsStaticSources",
			assertResult: func(t *testing.T, result []InputSource, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Len(t, result, 2)

				assert.Equal(t, "lena", result[0].ID)
				assert.Equal(t, "Lena", result[0].DisplayName)
				assert.Equal(t, "static", result[0].Type)
				assert.Equal(t, "/data/static_images/lena.png", result[0].ImagePath)
				assert.True(t, result[0].IsDefault)

				assert.Equal(t, "webcam", result[1].ID)
				assert.Equal(t, "Camera", result[1].DisplayName)
				assert.Equal(t, "camera", result[1].Type)
				assert.Empty(t, result[1].ImagePath)
				assert.False(t, result[1].IsDefault)
			},
		},
		{
			name: "Success_VerifySourceCounts",
			assertResult: func(t *testing.T, result []InputSource, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)

				staticCount := 0
				cameraCount := 0
				defaultCount := 0

				for _, src := range result {
					if src.Type == "static" {
						staticCount++
					}
					if src.Type == "camera" {
						cameraCount++
					}
					if src.IsDefault {
						defaultCount++
					}
				}

				assert.Equal(t, 1, staticCount)
				assert.Equal(t, 1, cameraCount)
				assert.Equal(t, 1, defaultCount)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockRepo := new(MockVideoRepository)
			mockRepo.On("List", mock.Anything).Return([]domain.Video{}, nil)
			sut := NewListInputsUseCase(mockRepo)
			ctx := context.Background()

			// Act
			result, err := sut.Execute(ctx)

			// Assert
			tt.assertResult(t, result, err)
			mockRepo.AssertExpectations(t)
		})
	}
}
