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

func (m *mockImageProcessor) ProcessImage(ctx context.Context, img *domain.Image, filters []domain.FilterType, accelerator domain.AcceleratorType, grayscaleType domain.GrayscaleType, blurParams *domain.BlurParameters) (*domain.Image, error) {
	args := m.Called(ctx, img, filters, accelerator, grayscaleType, blurParams)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*domain.Image), args.Error(1)
}

func makeRGBImage(width, height int) *domain.Image {
	return &domain.Image{
		Width:  width,
		Height: height,
		Data:   make([]byte, width*height*3),
	}
}

func makeGrayscaleImage(width, height int) *domain.Image {
	return &domain.Image{
		Width:  width,
		Height: height,
		Data:   make([]byte, width*height),
	}
}

func makeEmptyImage() *domain.Image {
	return &domain.Image{
		Width:  0,
		Height: 0,
		Data:   nil,
	}
}
func TestNewProcessImageUseCase(t *testing.T) {
	// Arrange
	processor := new(mockImageProcessor)

	// Act
	sut := NewProcessImageUseCase(processor)

	// Assert
	require.NotNil(t, sut, "expected use case to be created")
	assert.Equal(t, processor, sut.processor, "expected processor to be set correctly")
}

func TestProcessImageUseCase_Execute(t *testing.T) {
	var (
		errProcessingFailed = errors.New("processing failed")
		errInvalidImageData = errors.New("invalid image data")
	)

	tests := []struct {
		name          string
		inputImage    *domain.Image
		filters       []domain.FilterType
		accelerator   domain.AcceleratorType
		grayscaleType domain.GrayscaleType
		mockResult    *domain.Image
		mockError     error
		assertResult  func(t *testing.T, result *domain.Image, err error)
	}{
		{
			name:          "Success_CPUGrayscale",
			inputImage:    makeRGBImage(100, 100),
			filters:       []domain.FilterType{domain.FilterGrayscale},
			accelerator:   domain.AcceleratorCPU,
			grayscaleType: domain.GrayscaleAverage,
			mockResult:    makeGrayscaleImage(100, 100),
			mockError:     nil,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, 100, result.Width, "expected width 100")
				assert.Equal(t, 100, result.Height, "expected height 100")
			},
		},
		{
			name:          "Success_GPULuminosity",
			inputImage:    makeRGBImage(200, 150),
			filters:       []domain.FilterType{domain.FilterGrayscale},
			accelerator:   domain.AcceleratorGPU,
			grayscaleType: domain.GrayscaleLuminosity,
			mockResult:    makeGrayscaleImage(200, 150),
			mockError:     nil,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.Equal(t, 200, result.Width, "expected width 200")
			},
		},
		{
			name:          "Success_BT709Algorithm",
			inputImage:    makeRGBImage(100, 100),
			filters:       []domain.FilterType{domain.FilterGrayscale},
			accelerator:   domain.AcceleratorCPU,
			grayscaleType: domain.GrayscaleBT709,
			mockResult:    makeGrayscaleImage(100, 100),
			mockError:     nil,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.NoError(t, err)
				assert.NotNil(t, result, "expected non-nil result")
			},
		},
		{
			name:          "Error_ProcessingFailed",
			inputImage:    makeRGBImage(100, 100),
			filters:       []domain.FilterType{domain.FilterGrayscale},
			accelerator:   domain.AcceleratorCPU,
			grayscaleType: domain.GrayscaleAverage,
			mockResult:    nil,
			mockError:     errProcessingFailed,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.ErrorIs(t, err, errProcessingFailed)
				assert.Nil(t, result)
			},
		},
		{
			name:          "Edge_EmptyFilters",
			inputImage:    makeRGBImage(100, 100),
			filters:       []domain.FilterType{},
			accelerator:   domain.AcceleratorCPU,
			grayscaleType: domain.GrayscaleAverage,
			mockResult:    makeRGBImage(100, 100),
			mockError:     nil,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.NoError(t, err)
				require.NotNil(t, result)
				assert.NotEmpty(t, result.Data, "expected data to be present")
			},
		},
		{
			name:          "Error_NilImage",
			inputImage:    makeEmptyImage(),
			filters:       []domain.FilterType{domain.FilterGrayscale},
			accelerator:   domain.AcceleratorCPU,
			grayscaleType: domain.GrayscaleAverage,
			mockResult:    nil,
			mockError:     errInvalidImageData,
			assertResult: func(t *testing.T, result *domain.Image, err error) {
				assert.ErrorIs(t, err, errInvalidImageData)
				assert.Nil(t, result)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			mockProc := new(mockImageProcessor)
			mockProc.On("ProcessImage",
				mock.Anything,
				tt.inputImage,
				tt.filters,
				tt.accelerator,
				tt.grayscaleType,
				mock.Anything, // blurParams
			).Return(tt.mockResult, tt.mockError).Once()

			sut := NewProcessImageUseCase(mockProc)
			ctx := context.Background()

			// Act
			result, err := sut.Execute(ctx, tt.inputImage, tt.filters, tt.accelerator, tt.grayscaleType, nil)

			// Assert
			tt.assertResult(t, result, err)
			mockProc.AssertExpectations(t)
		})
	}
}

func TestProcessImageUseCase_Execute_ContextCancellation(t *testing.T) {
	// Arrange
	mockProc := new(mockImageProcessor)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	img := makeRGBImage(100, 100)

	mockProc.On("ProcessImage",
		mock.Anything,
		img,
		[]domain.FilterType{domain.FilterGrayscale},
		domain.AcceleratorCPU,
		domain.GrayscaleAverage,
		mock.Anything, // blurParams
	).Return(nil, ctx.Err()).Once()

	sut := NewProcessImageUseCase(mockProc)

	// Act
	result, err := sut.Execute(ctx, img, []domain.FilterType{domain.FilterGrayscale}, domain.AcceleratorCPU, domain.GrayscaleAverage, nil)

	// Assert
	assert.Error(t, err, "expected error from canceled context")
	assert.Nil(t, result, "expected nil result on context cancellation")
	mockProc.AssertExpectations(t)
}
