package video

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
)

type GoVideoPlayer struct {
	videoPath  string
	frameCount int
	duration   time.Duration
	stopped    bool
}

func NewGoVideoPlayer(videoPath string) (*GoVideoPlayer, error) {
	_, err := os.Stat(videoPath)
	if err != nil {
		return nil, fmt.Errorf("video file not found: %w", err)
	}

	return &GoVideoPlayer{
		videoPath:  videoPath,
		frameCount: 0,
		duration:   0,
		stopped:    false,
	}, nil
}

func (p *GoVideoPlayer) Play(ctx context.Context, videoPath string, frameCallback domain.FrameCallback) error {
	tracer := otel.Tracer("video-player")
	ctx, span := tracer.Start(ctx, "GoVideoPlayer.Play")
	defer span.End()

	span.SetAttributes(
		attribute.String("video.path", videoPath),
	)

	p.stopped = false

	frameNumber := 0
	framesProcessed := 0
	framesDropped := 0
	startTime := time.Now()

	span.AddEvent("Starting frame extraction")

	for !p.stopped {
		select {
		case <-ctx.Done():
			span.SetAttributes(
				attribute.Int("frames.processed", framesProcessed),
				attribute.Int("frames.dropped", framesDropped),
			)
			return ctx.Err()
		default:
		}

		frameStart := time.Now()

		frameData, err := p.extractFrame(ctx, frameNumber)
		if err != nil {
			if err.Error() == "end of video" {
				span.AddEvent("End of video reached, looping")
				frameNumber = 0
				continue
			}
			framesDropped++
			span.AddEvent("Frame extraction failed")
			continue
		}

		timestamp := time.Since(startTime)

		err = frameCallback(frameData, frameNumber, timestamp)
		if err != nil {
			framesDropped++
			span.AddEvent("Frame callback failed")
		} else {
			framesProcessed++
		}

		frameNumber++
		p.frameCount = frameNumber

		frameElapsed := time.Since(frameStart)
		if frameElapsed.Milliseconds() > 100 {
			span.AddEvent("Slow frame processing detected")
		}

		if framesProcessed > 0 && framesProcessed%30 == 0 {
			span.AddEvent("Playback progress")
		}
	}

	span.SetAttributes(
		attribute.Int("frames.processed.total", framesProcessed),
		attribute.Int("frames.dropped.total", framesDropped),
		attribute.Float64("playback.duration_seconds", time.Since(startTime).Seconds()),
	)

	return nil
}

func (p *GoVideoPlayer) Stop(ctx context.Context) error {
	tracer := otel.Tracer("video-player")
	_, span := tracer.Start(ctx, "GoVideoPlayer.Stop")
	defer span.End()

	p.stopped = true
	span.AddEvent("Video playback stopped")

	return nil
}

func (p *GoVideoPlayer) GetFrameCount() int {
	return p.frameCount
}

func (p *GoVideoPlayer) GetDuration() time.Duration {
	return p.duration
}

func (p *GoVideoPlayer) extractFrame(ctx context.Context, frameNumber int) ([]byte, error) {
	// TODO: Implement actual frame extraction using FFmpeg or Go video library
	// Currently this is a stub that always returns error, making Play() loop immediately
	return nil, fmt.Errorf("end of video")
}
