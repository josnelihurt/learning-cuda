package config

import (
	"context"
	"log"

	flipt "go.flipt.io/flipt-client"
)

type ObservabilityConfig struct {
	enabled               bool
	ServiceName           string
	ServiceVersion        string
	OtelCollectorEndpoint string
	TraceSamplingRate     float64
	fliptClient           fliptClientInterface
}

func (o *ObservabilityConfig) IsObservabilityEnabled(ctx context.Context) bool {
	return true //TODO: fixme
	enabled, err := o.fliptClient.EvaluateBoolean(ctx, &flipt.EvaluationRequest{
		FlagKey:  "enable_observability",
		EntityID: "observability",
	})
	if err != nil {
		log.Printf("Flipt evaluation failed for '%s': %v. Using YAML fallback.", "enable_observability", err)
		return o.enabled
	}
	return enabled.Enabled
}
