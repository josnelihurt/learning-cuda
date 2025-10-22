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

func evaluate[T any](
	ctx context.Context,
	operationName string,
	flagKey string,
	entityID string,
	fallbackValue T,
	evaluator func(context.Context, string, string) (*domain.FeatureFlagEvaluation, error),
	attributeSetter func(span trace.Span, value T),
) (T, error) {
	tracer := otel.Tracer("evaluate-feature-flag")
	ctx, span := tracer.Start(ctx, operationName,
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("flag.key", flagKey),
		attribute.String("flag.entity_id", entityID),
	)
	attributeSetter(span, fallbackValue)

	eval, err := evaluator(ctx, flagKey, entityID)
	if err != nil || !eval.Success {
		log.Printf("Feature flag '%s' evaluation failed, using fallback: %v. Error: %v",
			flagKey, fallbackValue, err)
		span.SetAttributes(attribute.Bool("flag.used_fallback", true))
		attributeSetter(span, fallbackValue)
		return fallbackValue, nil
	}

	result, ok := eval.Result.(T)
	if !ok {
		log.Printf("Warning: Type assertion failed for flag '%s', result type: %T, using fallback value", flagKey, eval.Result)
		span.SetAttributes(attribute.Bool("flag.used_fallback", true))
		attributeSetter(span, fallbackValue)
		return fallbackValue, nil
	}
	span.SetAttributes(attribute.Bool("flag.used_fallback", false))
	attributeSetter(span, result)

	log.Printf("Feature flag '%s' evaluated to: %v", flagKey, result)
	return result, nil
}

func (uc *EvaluateFeatureFlagUseCase) EvaluateBoolean(
	ctx context.Context,
	flagKey string,
	entityID string,
	fallbackValue bool,
) (bool, error) {
	return evaluate(
		ctx, "EvaluateBoolean", flagKey, entityID, fallbackValue,
		uc.repository.EvaluateBoolean,
		func(span trace.Span, value bool) {
			span.SetAttributes(attribute.Bool("flag.fallback_value", value))
		},
	)
}

func (uc *EvaluateFeatureFlagUseCase) EvaluateVariant(
	ctx context.Context,
	flagKey string,
	entityID string,
	fallbackValue string,
) (string, error) {
	return evaluate(
		ctx, "EvaluateVariant", flagKey, entityID, fallbackValue,
		uc.repository.EvaluateVariant,
		func(span trace.Span, value string) {
			span.SetAttributes(attribute.String("flag.fallback_value", value))
		},
	)
}
