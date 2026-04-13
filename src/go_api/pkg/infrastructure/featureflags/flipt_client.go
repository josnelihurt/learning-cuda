package featureflags

import (
	"context"

	flipt "go.flipt.io/flipt-client"
)

type FliptClientInterface interface {
	EvaluateBoolean(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.BooleanEvaluationResponse, error)
	EvaluateString(ctx context.Context, req *flipt.EvaluationRequest) (*flipt.VariantEvaluationResponse, error)
	Close(ctx context.Context) error
}
