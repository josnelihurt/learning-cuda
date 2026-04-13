package application

import (
	"context"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ListAvailableImagesUseCase struct {
	repository domain.StaticImageRepository
}

func NewListAvailableImagesUseCase(repository domain.StaticImageRepository) *ListAvailableImagesUseCase {
	return &ListAvailableImagesUseCase{
		repository: repository,
	}
}

func (uc *ListAvailableImagesUseCase) Execute(ctx context.Context) ([]domain.StaticImage, error) {
	tracer := otel.Tracer("list-available-images")
	ctx, span := tracer.Start(ctx, "ListAvailableImagesUseCase.Execute",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	images, err := uc.repository.FindAll(ctx)
	if err != nil {
		span.RecordError(err)
		return nil, err
	}

	span.SetAttributes(
		attribute.Int("images.count", len(images)),
	)

	return images, nil
}
