package application

import (
	"context"
	"fmt"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type VideoPlaybackUseCase struct {
	videoRepository domain.VideoRepository
	videoPlayer     domain.VideoPlayer
	imageProcessor  *ProcessImageUseCase
}

func NewVideoPlaybackUseCase(
	videoRepository domain.VideoRepository,
	videoPlayer domain.VideoPlayer,
	imageProcessor *ProcessImageUseCase,
) *VideoPlaybackUseCase {
	return &VideoPlaybackUseCase{
		videoRepository: videoRepository,
		videoPlayer:     videoPlayer,
		imageProcessor:  imageProcessor,
	}
}

func (uc *VideoPlaybackUseCase) Execute(
	ctx context.Context,
	videoID string,
	filters []domain.FilterType,
	accelerator domain.AcceleratorType,
	grayscaleType domain.GrayscaleType,
	frameCallback func(processedData []byte, frameNumber int) error,
) error {
	tracer := otel.Tracer("video-playback")
	ctx, span := tracer.Start(ctx, "VideoPlayback",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("video.id", videoID),
		attribute.Int("filters.count", len(filters)),
		attribute.String("accelerator", string(accelerator)),
	)

	video, err := uc.videoRepository.GetByID(ctx, videoID)
	if err != nil {
		span.SetAttributes(attribute.Bool("error.video_not_found", true))
		return fmt.Errorf("video not found: %w", err)
	}

	framesProcessed := 0
	framesDropped := 0
	startTime := time.Now()

	callback := func(frameData []byte, frameNumber int, timestamp time.Duration) error {
		if len(frameData) == 0 {
			framesDropped++
			return nil
		}

		framesProcessed++

		if framesProcessed%30 == 0 {
			// TODO: Add FPS metric to span attributes instead of discarding calculation
			_ = float64(framesProcessed) / time.Since(startTime).Seconds()
			span.AddEvent("Playback metrics")
		}

		return frameCallback(frameData, frameNumber)
	}

	err = uc.videoPlayer.Play(ctx, video.Path, callback)
	if err != nil && err != context.Canceled {
		span.SetAttributes(attribute.Bool("error.playback_failed", true))
		return fmt.Errorf("playback failed: %w", err)
	}

	span.SetAttributes(
		attribute.Int("frames.processed.total", framesProcessed),
		attribute.Int("frames.dropped.total", framesDropped),
	)

	return nil
}
