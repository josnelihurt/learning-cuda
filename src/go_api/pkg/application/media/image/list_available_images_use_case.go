package image

import (
	"context"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ListAvailableImagesUseCaseInput struct {
}

type ListAvailableImagesUseCaseOutput struct {
	Images []domain.StaticImage
}

type ListAvailableImagesUseCase struct {
	repository staticImageRepository
}

func NewListAvailableImagesUseCase(repository staticImageRepository) *ListAvailableImagesUseCase {
	return &ListAvailableImagesUseCase{
		repository: repository,
	}
}

func (uc *ListAvailableImagesUseCase) Execute(ctx context.Context, input ListAvailableImagesUseCaseInput) (ListAvailableImagesUseCaseOutput, error) {
	tracer := otel.Tracer("list-available-images")
	ctx, span := tracer.Start(ctx, "ListAvailableImagesUseCase.Execute",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	images, err := uc.repository.FindAll(ctx)
	if err != nil {
		span.RecordError(err)
		return ListAvailableImagesUseCaseOutput{}, err
	}

	span.SetAttributes(
		attribute.Int("images.count", len(images)),
	)

	return ListAvailableImagesUseCaseOutput{
		Images: images,
	}, nil
}
