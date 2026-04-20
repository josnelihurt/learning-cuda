package image

import (
	"context"
	"fmt"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ProcessImageUseCaseInput struct {
	Image *domain.Image
	Opts  domain.ProcessingOptions
}

type ProcessImageUseCaseOutput struct {
	Image *domain.Image
}

type ProcessImageUseCase struct {
	processor imageProcessor
}

func NewProcessImageUseCase(processor imageProcessor) *ProcessImageUseCase {
	return &ProcessImageUseCase{
		processor: processor,
	}
}

func (uc *ProcessImageUseCase) Execute(ctx context.Context, input ProcessImageUseCaseInput) (ProcessImageUseCaseOutput, error) {
	tracer := otel.Tracer("process-image-use-case")
	ctx, span := tracer.Start(ctx, "ProcessImageUseCase.Execute",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.Int("image.width", input.Image.Width),
		attribute.Int("image.height", input.Image.Height),
		attribute.Int("image.size_bytes", len(input.Image.Data)),
		attribute.String("accelerator", string(input.Opts.Accelerator)),
		attribute.String("grayscale_type", string(input.Opts.GrayscaleType)),
		attribute.Int("filters_count", len(input.Opts.Filters)),
	)

	filterNames := make([]string, len(input.Opts.Filters))
	for i, f := range input.Opts.Filters {
		filterNames[i] = string(f)
	}
	span.SetAttributes(attribute.StringSlice("filters", filterNames))

	if input.Opts.BlurParams != nil {
		span.SetAttributes(
			attribute.Int("blur.kernel_size", int(input.Opts.BlurParams.KernelSize)),
			attribute.Float64("blur.sigma", float64(input.Opts.BlurParams.Sigma)),
			attribute.String("blur.border_mode", string(input.Opts.BlurParams.BorderMode)),
			attribute.Bool("blur.separable", input.Opts.BlurParams.Separable),
		)
	}

	result, err := uc.processor.ProcessImage(ctx, input.Image, input.Opts)
	if err != nil {
		span.RecordError(err)
		return ProcessImageUseCaseOutput{}, fmt.Errorf("failed to process image: %w", err)
	}

	span.SetAttributes(
		attribute.Int("result.width", result.Width),
		attribute.Int("result.height", result.Height),
		attribute.Int("result.size_bytes", len(result.Data)),
	)

	return ProcessImageUseCaseOutput{Image: result}, nil
}
