package config

import (
	"context"

	flipt "go.flipt.io/flipt-client"
)

// FliptClientInterface defines the interface for feature flag evaluation
type FliptClientInterface interface {
	EvaluateBoolean(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.BooleanEvaluationResponse, error)
	EvaluateString(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.VariantEvaluationResponse, error)
	Close(ctx context.Context) error
}

type fliptClientInterface = FliptClientInterface
