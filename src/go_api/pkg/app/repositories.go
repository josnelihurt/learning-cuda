package app

import (
	"context"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
)

type featureFlagRepository interface {
	EvaluateBoolean(ctx context.Context, flagKey, entityID string) (*domain.FeatureFlagEvaluation, error)
	EvaluateString(ctx context.Context, flagKey, entityID string) (*domain.FeatureFlagEvaluation, error)
	GetFlag(ctx context.Context, flagKey string) (*domain.FeatureFlag, error)
	ListFlags(ctx context.Context) ([]domain.FeatureFlag, error)
	UpsertFlag(ctx context.Context, flag domain.FeatureFlag) error
}
