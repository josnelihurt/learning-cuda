package video

import (
	"context"
	"time"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
)

type StreamVideoPlayer interface {
	Play(ctx context.Context, frameCallback func(*domain.Image, int, time.Duration) error) error
}

type StreamVideoPlayerFactory func(videoPath string) (StreamVideoPlayer, error)

type StreamVideoPeer interface {
	Connect(ctx context.Context) error
	Send(payload []byte) error
	Close() error
	Label() string
}

type StreamVideoPeerFactory func(browserSessionID string) (StreamVideoPeer, error)
