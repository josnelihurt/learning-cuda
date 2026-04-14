package domain

import (
	"context"
	"time"
)

type FrameCallback func(frameData []byte, frameNumber int, timestamp time.Duration) error

type VideoPlayer interface {
	Play(ctx context.Context, videoPath string, frameCallback FrameCallback) error
	Stop(ctx context.Context) error
	GetFrameCount() int
	GetDuration() time.Duration
}
