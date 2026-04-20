package video

import (
	"context"
	"time"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
)

type videoRepository interface {
	List(ctx context.Context) ([]domain.Video, error)
	GetByID(ctx context.Context, id string) (*domain.Video, error)
	Save(ctx context.Context, video *domain.Video) error
}

type videoPlayer interface {
	Play(ctx context.Context, videoPath string, frameCallback domain.FrameCallback) error
	Stop(ctx context.Context) error
	GetFrameCount() int
	GetDuration() time.Duration
}
