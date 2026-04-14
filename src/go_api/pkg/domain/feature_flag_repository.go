package domain

import "context"

type FeatureFlagRepository interface {
	EvaluateBoolean(ctx context.Context, flagKey, entityID string) (*FeatureFlagEvaluation, error)
	EvaluateString(ctx context.Context, flagKey, entityID string) (*FeatureFlagEvaluation, error)
	GetFlag(ctx context.Context, flagKey string) (*FeatureFlag, error)
	ListFlags(ctx context.Context) ([]FeatureFlag, error)
	UpsertFlag(ctx context.Context, flag FeatureFlag) error
}
