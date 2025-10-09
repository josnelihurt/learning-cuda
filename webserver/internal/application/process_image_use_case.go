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

// Execute processes an image with the specified filter
func (uc *ProcessImageUseCase) Execute(img *domain.Image, filter domain.FilterType) (*domain.Image, error) {
	return uc.processor.ProcessImage(img, filter)
}

