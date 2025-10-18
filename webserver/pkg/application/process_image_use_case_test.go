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

type mockImageProcessor struct {
	mock.Mock
}

func (m *mockImageProcessor) ProcessImage(ctx context.Context, img *domain.Image, filters []domain.FilterType, accelerator domain.AcceleratorType, grayscaleType domain.GrayscaleType) (*domain.Image, error) {
	args := m.Called(ctx, img, filters, accelerator, grayscaleType)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.Image), args.Error(1)
}

func makeValidImage() *domain.Image {
	return &domain.Image{
		Data:   []byte{1, 2, 3, 4},
		Width:  100,
		Height: 100,
		Format: "png",
	}
}

func makeProcessedImage() *domain.Image {
	return &domain.Image{
		Data:   []byte{5, 6, 7, 8},
		Width:  100,
		Height: 100,
		Format: "png",
	}
}

func makeValidFilters() []domain.FilterType {
	return []domain.FilterType{
		domain.FilterGrayscale,
	}
}

func makeEmptyFilters() []domain.FilterType {
	return []domain.FilterType{}
}

func TestNewProcessImageUseCase(t *testing.T) {
	// Arrange
	processor := new(mockImageProcessor)

	// Act
	sut := NewProcessImageUseCase(processor)

	// Assert
	require.NotNil(t, sut)
	assert.Equal(t, processor, sut.processor)
}

func TestProcessImageUseCase_Execute(t *testing.T) {
	var (
		errProcessingFailed = errors.New("processing failed")
	)

	tests := []struct {
		name         string
		img          *domain.Image
		filters      []domain.FilterType
		accelerator  domain.AcceleratorType
		grayscaleType domain.GrayscaleType
		mockResult   *domain.Image
		mockError    error
		assertResult func(t *testing.T, result *domain.Image, err error)
	}{
		{
			name:          "Success_ValidProcessing",
			img:           makeValidImage(),
			filters:       makeValidFilters(),
			accelerator:   domain.AcceleratorGPU,
			grayscaleType: domain.GrayscaleBT709,
			mockResult:    makeProcessedImage(),
			mockError:     nil,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, 100, result.Width)
				assert.Equal(t, 100, result.Height)
				assert.Equal(t, "png", result.Format)
				assert.Equal(t, []byte{5, 6, 7, 8}, result.Data)
			},
		},
		{
			name:          "Success_NoFilters",
			img:           makeValidImage(),
			filters:       makeEmptyFilters(),
			accelerator:   domain.AcceleratorCPU,
			grayscaleType: domain.GrayscaleAverage,
			mockResult:    makeProcessedImage(),
			mockError:     nil,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, 100, result.Width)
				assert.Equal(t, 100, result.Height)
			},
		},
		{
			name:          "Error_ProcessingFailed",
			img:           makeValidImage(),
			filters:       makeValidFilters(),
			accelerator:   domain.AcceleratorGPU,
			grayscaleType: domain.GrayscaleBT709,
			mockResult:    nil,
			mockError:     errProcessingFailed,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "failed to process image")
				assert.Nil(t, result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockProcessor := new(mockImageProcessor)
			mockProcessor.On("ProcessImage",
				mock.Anything,
				tt.img,
				tt.filters,
				tt.accelerator,
				tt.grayscaleType,
			).Return(tt.mockResult, tt.mockError).Once()

			sut := NewProcessImageUseCase(mockProcessor)
			ctx := context.Background()

			// Act
			result, err := sut.Execute(ctx, tt.img, tt.filters, tt.accelerator, tt.grayscaleType)

			// Assert
			tt.assertResult(t, result, err)
			mockProcessor.AssertExpectations(t)
		})
	}
}

func TestProcessImageUseCase_ContextCancellation(t *testing.T) {
	// Arrange
	mockProcessor := new(mockImageProcessor)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mockProcessor.On("ProcessImage",
		mock.Anything,
		mock.Anything,
		mock.Anything,
		mock.Anything,
		mock.Anything,
	).Return(nil, ctx.Err()).Once()

	sut := NewProcessImageUseCase(mockProcessor)

	// Act
	result, err := sut.Execute(ctx, makeValidImage(), makeValidFilters(), domain.AcceleratorGPU, domain.GrayscaleBT709)

	// Assert
	assert.Error(t, err)
	assert.Nil(t, result)
	mockProcessor.AssertExpectations(t)
}