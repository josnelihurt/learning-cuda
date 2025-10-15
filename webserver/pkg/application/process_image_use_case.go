package application

import (
	"context"
	"fmt"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
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
func (uc *ProcessImageUseCase) Execute(ctx context.Context, img *domain.Image, filters []domain.FilterType, accelerator domain.AcceleratorType, grayscaleType domain.GrayscaleType) (*domain.Image, error) {
	tracer := otel.Tracer("process-image-use-case")
	ctx, span := tracer.Start(ctx, "ProcessImageUseCase.Execute",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.Int("image.width", img.Width),
		attribute.Int("image.height", img.Height),
		attribute.Int("image.size_bytes", len(img.Data)),
		attribute.String("accelerator", string(accelerator)),
		attribute.String("grayscale_type", string(grayscaleType)),
		attribute.Int("filters_count", len(filters)),
	)

	filterNames := make([]string, len(filters))
	for i, f := range filters {
		filterNames[i] = string(f)
	}
	span.SetAttributes(attribute.StringSlice("filters", filterNames))

	result, err := uc.processor.ProcessImage(ctx, img, filters, accelerator, grayscaleType)
	if err != nil {
		span.RecordError(err)
		return nil, fmt.Errorf("failed to process image: %w", err)
	}

	span.SetAttributes(
		attribute.Int("result.width", result.Width),
		attribute.Int("result.height", result.Height),
		attribute.Int("result.size_bytes", len(result.Data)),
	)

	return result, nil
}

