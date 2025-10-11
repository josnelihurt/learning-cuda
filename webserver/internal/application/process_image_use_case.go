package application

import (
	"github.com/jrb/cuda-learning/webserver/internal/domain"
)

// ProcessImageUseCase handles the business logic for image processing
type ProcessImageUseCase struct {
	processor domain.ImageProcessor
}

// NewProcessImageUseCase creates a new use case instance
func NewProcessImageUseCase(processor domain.ImageProcessor) *ProcessImageUseCase {
	return &ProcessImageUseCase{
		processor: processor,
	}
}

// Execute processes an image with the specified filters, accelerator, and grayscale type
func (uc *ProcessImageUseCase) Execute(img *domain.Image, filters []domain.FilterType, accelerator domain.AcceleratorType, grayscaleType domain.GrayscaleType) (*domain.Image, error) {
	return uc.processor.ProcessImage(img, filters, accelerator, grayscaleType)
}

