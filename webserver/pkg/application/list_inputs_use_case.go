package application

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type InputSource struct {
	ID          string
	DisplayName string
	Type        string
	ImagePath   string
	IsDefault   bool
}

type ListInputsUseCase struct{}

func NewListInputsUseCase() *ListInputsUseCase {
	return &ListInputsUseCase{}
}

func (uc *ListInputsUseCase) Execute(ctx context.Context) ([]InputSource, error) {
	tracer := otel.Tracer("list-inputs")
	_, span := tracer.Start(ctx, "ListInputs",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	// TODO: Load static images dynamically from /data directory scan
	// TODO: Support multiple connected cameras (server-side cameras)
	// TODO: Support stored video files as input sources
	// TODO: Support remote stream URLs (RTSP, HLS, etc.)
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

	staticCount := 0
	cameraCount := 0
	for _, src := range sources {
		if src.Type == "static" {
			staticCount++
		} else if src.Type == "camera" {
			cameraCount++
		}
	}

	span.SetAttributes(
		attribute.Int("input_sources.count", len(sources)),
		attribute.Int("input_sources.static_count", staticCount),
		attribute.Int("input_sources.camera_count", cameraCount),
	)

	return sources, nil
}
