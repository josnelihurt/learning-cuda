package application

import (
	"context"
	"log"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type EvaluateFeatureFlagUseCase struct {
	repository domain.FeatureFlagRepository
}

func NewEvaluateFeatureFlagUseCase(repo domain.FeatureFlagRepository) *EvaluateFeatureFlagUseCase {
	return &EvaluateFeatureFlagUseCase{repository: repo}
}

func (uc *EvaluateFeatureFlagUseCase) EvaluateBoolean(
	ctx context.Context,
	flagKey string,
	entityID string,
	fallbackValue bool,
) (bool, error) {
	tracer := otel.Tracer("evaluate-feature-flag")
	ctx, span := tracer.Start(ctx, "EvaluateBoolean",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("flag.key", flagKey),
		attribute.String("flag.entity_id", entityID),
		attribute.Bool("flag.fallback_value", fallbackValue),
	)

	eval, err := uc.repository.EvaluateBoolean(ctx, flagKey, entityID)
	if err != nil || !eval.Success {
		log.Printf("Feature flag '%s' evaluation failed, using fallback: %v. Error: %v", flagKey, fallbackValue, err)
		span.SetAttributes(attribute.Bool("flag.used_fallback", true))
		span.SetAttributes(attribute.Bool("flag.result", fallbackValue))
		return fallbackValue, nil
	}

	result := eval.Result.(bool)
	span.SetAttributes(
		attribute.Bool("flag.result", result),
		attribute.Bool("flag.used_fallback", false),
	)

	log.Printf("Feature flag '%s' evaluated to: %v", flagKey, result)
	return result, nil
}

func (uc *EvaluateFeatureFlagUseCase) EvaluateVariant(
	ctx context.Context,
	flagKey string,
	entityID string,
	fallbackValue string,
) (string, error) {
	tracer := otel.Tracer("evaluate-feature-flag")
	ctx, span := tracer.Start(ctx, "EvaluateVariant",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("flag.key", flagKey),
		attribute.String("flag.entity_id", entityID),
		attribute.String("flag.fallback_value", fallbackValue),
	)

	eval, err := uc.repository.EvaluateVariant(ctx, flagKey, entityID)
	if err != nil || !eval.Success {
		log.Printf("Feature flag '%s' evaluation failed, using fallback: %s. Error: %v", flagKey, fallbackValue, err)
		span.SetAttributes(attribute.Bool("flag.used_fallback", true))
		span.SetAttributes(attribute.String("flag.result", fallbackValue))
		return fallbackValue, nil
	}

	result := eval.Result.(string)
	span.SetAttributes(
		attribute.String("flag.result", result),
		attribute.Bool("flag.used_fallback", false),
	)

	log.Printf("Feature flag '%s' evaluated to: %s", flagKey, result)
	return result, nil
}
