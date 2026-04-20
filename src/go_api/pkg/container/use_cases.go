package container

import "context"

type useCase[Input any, Output any] interface {
	Execute(ctx context.Context, input Input) (Output, error)
}

type evaluateFeatureFlagUseCase interface {
	EvaluateBoolean(ctx context.Context, flagKey string, entityID string, fallbackValue bool) (bool, error)
	EvaluateString(ctx context.Context, flagKey string, entityID string, fallbackValue string) (string, error)
}
