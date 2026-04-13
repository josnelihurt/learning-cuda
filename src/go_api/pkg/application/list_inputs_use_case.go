package application

import (
	"context"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type InputSource struct {
	ID               string
	DisplayName      string
	Type             string
	ImagePath        string
	IsDefault        bool
	VideoPath        string
	PreviewImagePath string
}

type ListInputsUseCase struct {
	videoRepository domain.VideoRepository
}

func NewListInputsUseCase(videoRepository domain.VideoRepository) *ListInputsUseCase {
	return &ListInputsUseCase{
		videoRepository: videoRepository,
	}
}

func (uc *ListInputsUseCase) Execute(ctx context.Context) ([]InputSource, error) {
	tracer := otel.Tracer("list-inputs")
	_, span := tracer.Start(ctx, "ListInputs",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	sources := []InputSource{
		{
			ID:          "lena",
			DisplayName: "Lena",
			Type:        "static",
			ImagePath:   "/data/static_images/lena.png",
			IsDefault:   true,
		},
		{
			ID:          "webcam",
			DisplayName: "Camera",
			Type:        "camera",
			ImagePath:   "",
			IsDefault:   false,
		},
	}

	videos, err := uc.videoRepository.List(ctx)
	if err == nil {
		for _, vid := range videos {
			sources = append(sources, InputSource{
				ID:               vid.ID,
				DisplayName:      vid.DisplayName,
				Type:             "video",
				VideoPath:        vid.Path,
				PreviewImagePath: vid.PreviewImagePath,
				IsDefault:        vid.IsDefault,
			})
		}
	}

	staticCount := 0
	cameraCount := 0
	videoCount := 0
	for _, src := range sources {
		switch src.Type {
		case "static":
			staticCount++
		case "camera":
			cameraCount++
		case "video":
			videoCount++
		}
	}

	span.SetAttributes(
		attribute.Int("input_sources.count", len(sources)),
		attribute.Int("input_sources.static_count", staticCount),
		attribute.Int("input_sources.camera_count", cameraCount),
		attribute.Int("input_sources.video_count", videoCount),
	)

	return sources, nil
}
