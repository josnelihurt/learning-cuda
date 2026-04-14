package application

import (
	"context"

	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/logger"
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
		logger.FromContext(ctx).Warn().
			Str("flag_key", flagKey).
			Bool("fallback_value", fallbackValue).
			Err(err).
			Msg("Feature flag evaluation failed, using fallback")
		span.SetAttributes(attribute.Bool("flag.used_fallback", true))
		span.SetAttributes(attribute.Bool("flag.fallback_value", fallbackValue))
		return fallbackValue, nil
	}

	result, ok := eval.Result.(bool)
	if !ok {
		logger.FromContext(ctx).Warn().
			Str("flag_key", flagKey).
			Msg("Type assertion failed for flag result, using fallback value")
		span.SetAttributes(attribute.Bool("flag.used_fallback", true))
		span.SetAttributes(attribute.Bool("flag.fallback_value", fallbackValue))
		return fallbackValue, nil
	}
	span.SetAttributes(attribute.Bool("flag.used_fallback", false))
	span.SetAttributes(attribute.Bool("flag.fallback_value", result))

	logger.FromContext(ctx).Debug().Str("flag_key", flagKey).Bool("result", result).Msg("Feature flag evaluated")
	return result, nil
}

func (uc *EvaluateFeatureFlagUseCase) EvaluateString(
	ctx context.Context,
	flagKey string,
	entityID string,
	fallbackValue string,
) (string, error) {
	tracer := otel.Tracer("evaluate-feature-flag")
	ctx, span := tracer.Start(ctx, "EvaluateString",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		attribute.String("flag.key", flagKey),
		attribute.String("flag.entity_id", entityID),
		attribute.String("flag.fallback_value", fallbackValue),
	)

	eval, err := uc.repository.EvaluateString(ctx, flagKey, entityID)
	if err != nil || !eval.Success {
		logger.FromContext(ctx).Warn().
			Str("flag_key", flagKey).
			Str("fallback_value", fallbackValue).
			Err(err).
			Msg("Feature flag evaluation failed, using fallback")
		span.SetAttributes(attribute.Bool("flag.used_fallback", true))
		span.SetAttributes(attribute.String("flag.fallback_value", fallbackValue))
		return fallbackValue, nil
	}

	result, ok := eval.Result.(string)
	if !ok {
		logger.FromContext(ctx).Warn().
			Str("flag_key", flagKey).
			Msg("Type assertion failed for flag result, using fallback value")
		span.SetAttributes(attribute.Bool("flag.used_fallback", true))
		span.SetAttributes(attribute.String("flag.fallback_value", fallbackValue))
		return fallbackValue, nil
	}
	span.SetAttributes(attribute.Bool("flag.used_fallback", false))
	span.SetAttributes(attribute.String("flag.fallback_value", result))

	logger.FromContext(ctx).Debug().Str("flag_key", flagKey).Str("result", result).Msg("Feature flag evaluated")
	return result, nil
}
