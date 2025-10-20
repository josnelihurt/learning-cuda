package application

import (
	"context"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ListVideosUseCase struct {
	repository domain.VideoRepository
}

func NewListVideosUseCase(repository domain.VideoRepository) *ListVideosUseCase {
	return &ListVideosUseCase{
		repository: repository,
	}
}

func (uc *ListVideosUseCase) Execute(ctx context.Context) ([]domain.Video, error) {
	tracer := otel.Tracer("list-videos")
	_, span := tracer.Start(ctx, "ListVideos",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	videos, err := uc.repository.List(ctx)
	if err != nil {
		span.SetAttributes(attribute.Bool("error", true))
		return nil, err
	}

	videoCount := 0
	defaultCount := 0
	for _, v := range videos {
		videoCount++
		if v.IsDefault {
			defaultCount++
		}
	}

	span.SetAttributes(
		attribute.Int("videos.count", videoCount),
		attribute.Int("videos.default_count", defaultCount),
	)

	return videos, nil
}
