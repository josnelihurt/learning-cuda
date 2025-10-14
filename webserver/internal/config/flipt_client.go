package config

import (
	"context"

	flipt "go.flipt.io/flipt-client"
)

// fliptClientInterface defines the interface for feature flag evaluation
type fliptClientInterface interface {
	EvaluateBoolean(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.BooleanEvaluationResponse, error)
	EvaluateString(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.VariantEvaluationResponse, error)
	Close(ctx context.Context) error
}

