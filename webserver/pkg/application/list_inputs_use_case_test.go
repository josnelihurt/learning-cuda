package application

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewListInputsUseCase(t *testing.T) {
	// Arrange & Act
	sut := NewListInputsUseCase()

	// Assert
	require.NotNil(t, sut)
}

func TestListInputsUseCase_Execute(t *testing.T) {
	tests := []struct {
		name         string
		assertResult func(t *testing.T, result []InputSource, err error)
	}{
		{
			name: "Success_ReturnsInputSources",
			assertResult: func(t *testing.T, result []InputSource, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Len(t, result, 2)

				// Check first source (lena)
				assert.Equal(t, "lena", result[0].ID)
				assert.Equal(t, "Lena", result[0].DisplayName)
				assert.Equal(t, "static", result[0].Type)
				assert.Equal(t, "/data/lena.png", result[0].ImagePath)
				assert.True(t, result[0].IsDefault)

				// Check second source (webcam)
				assert.Equal(t, "webcam", result[1].ID)
				assert.Equal(t, "Camera", result[1].DisplayName)
				assert.Equal(t, "camera", result[1].Type)
				assert.Equal(t, "", result[1].ImagePath)
				assert.False(t, result[1].IsDefault)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			sut := NewListInputsUseCase()
			ctx := context.Background()

			// Act
			result, err := sut.Execute(ctx)

			// Assert
			tt.assertResult(t, result, err)
		})
	}
}

func TestListInputsUseCase_ContextCancellation(t *testing.T) {
	// Arrange
	sut := NewListInputsUseCase()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Act
	result, err := sut.Execute(ctx)

	// Assert
	assert.NoError(t, err)
	require.NotNil(t, result)
	assert.Len(t, result, 2)
}