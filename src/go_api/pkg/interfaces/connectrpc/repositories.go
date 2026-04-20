package connectrpc

import (
	"context"

	"github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
)

type featureFlagRepository interface {
	EvaluateBoolean(ctx context.Context, flagKey, entityID string) (*domain.FeatureFlagEvaluation, error)
	EvaluateString(ctx context.Context, flagKey, entityID string) (*domain.FeatureFlagEvaluation, error)
	GetFlag(ctx context.Context, flagKey string) (*domain.FeatureFlag, error)
	ListFlags(ctx context.Context) ([]domain.FeatureFlag, error)
	UpsertFlag(ctx context.Context, flag domain.FeatureFlag) error
}

type listInputsUseCase interface {
	Execute(ctx context.Context) ([]video.InputSource, error)
}

type evaluateFeatureFlagUseCase interface {
	EvaluateBoolean(ctx context.Context, flagKey string, entityID string, fallbackValue bool) (bool, error)
	EvaluateString(ctx context.Context, flagKey string, entityID string, fallbackValue string) (string, error)
}
type getSystemInfoUseCase interface {
	Execute(ctx context.Context) (*domain.SystemInfo, error)
}
