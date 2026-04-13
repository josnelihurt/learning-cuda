package domain

import "context"

type FeatureFlagRepository interface {
	EvaluateBoolean(ctx context.Context, flagKey, entityID string) (*FeatureFlagEvaluation, error)
	EvaluateVariant(ctx context.Context, flagKey, entityID string) (*FeatureFlagEvaluation, error)
	SyncFlags(ctx context.Context, flags []FeatureFlag) error
	GetFlag(ctx context.Context, flagKey string) (*FeatureFlag, error)
}
