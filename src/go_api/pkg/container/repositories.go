package container

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
type featureFlagRepository interface {
	EvaluateBoolean(ctx context.Context, flagKey, entityID string) (*domain.FeatureFlagEvaluation, error)
	EvaluateString(ctx context.Context, flagKey, entityID string) (*domain.FeatureFlagEvaluation, error)
	GetFlag(ctx context.Context, flagKey string) (*domain.FeatureFlag, error)
	ListFlags(ctx context.Context) ([]domain.FeatureFlag, error)
	UpsertFlag(ctx context.Context, flag domain.FeatureFlag) error
}
