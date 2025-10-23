package featureflags

import (
	"context"
	"fmt"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	flipt "go.flipt.io/flipt-client"
)

type FliptRepository struct {
	reader FliptClientInterface
	writer *FliptWriter
}

func NewFliptRepository(reader FliptClientInterface, writer *FliptWriter) *FliptRepository {
	return &FliptRepository{
		reader: reader,
		writer: writer,
	}
}

// evaluateFlag is a generic helper for evaluating feature flags with common error handling
func (r *FliptRepository) evaluateFlag(
	ctx context.Context,
	flagKey, entityID string,
	evaluationType string,
	evaluateFunc func(context.Context, *flipt.EvaluationRequest) (interface{}, error),
	extractResult func(interface{}) interface{},
) (*domain.FeatureFlagEvaluation, error) {
	log := logger.FromContext(ctx)
	if r.reader == nil {
		log.Warn().
			Str("flag_key", flagKey).
			Msg("Flipt client not initialized, returning fallback evaluation")
		return &domain.FeatureFlagEvaluation{
			FlagKey:      flagKey,
			EntityID:     entityID,
			Success:      false,
			UsedFallback: true,
		}, nil
	}

	resp, err := evaluateFunc(ctx, &flipt.EvaluationRequest{
		FlagKey:  flagKey,
		EntityID: entityID,
	})

	if err != nil {
		log.Warn().
			Err(err).
			Str("flag_key", flagKey).
			Msgf("Flipt %s evaluation failed", evaluationType)
		return &domain.FeatureFlagEvaluation{
			FlagKey:      flagKey,
			EntityID:     entityID,
			Success:      false,
			UsedFallback: true,
		}, err
	}

	return &domain.FeatureFlagEvaluation{
		FlagKey:      flagKey,
		EntityID:     entityID,
		Result:       extractResult(resp),
		Success:      true,
		UsedFallback: false,
	}, nil
}

func (r *FliptRepository) EvaluateBoolean(
	ctx context.Context,
	flagKey, entityID string,
) (*domain.FeatureFlagEvaluation, error) {
	return r.evaluateFlag(
		ctx,
		flagKey,
		entityID,
		"boolean",
		func(ctx context.Context, req *flipt.EvaluationRequest) (interface{}, error) {
			return r.reader.EvaluateBoolean(ctx, req)
		},
		func(resp interface{}) interface{} {
			booleanResp, ok := resp.(*flipt.BooleanEvaluationResponse)
			if !ok {
				return false // fallback value
			}
			return booleanResp.Enabled
		},
	)
}

func (r *FliptRepository) EvaluateVariant(
	ctx context.Context,
	flagKey, entityID string,
) (*domain.FeatureFlagEvaluation, error) {
	return r.evaluateFlag(
		ctx,
		flagKey,
		entityID,
		"variant",
		func(ctx context.Context, req *flipt.EvaluationRequest) (interface{}, error) {
			return r.reader.EvaluateString(ctx, req)
		},
		func(resp interface{}) interface{} {
			variantResp, ok := resp.(*flipt.VariantEvaluationResponse)
			if !ok {
				return "" // fallback value
			}
			return variantResp.VariantAttachment
		},
	)
}

func (r *FliptRepository) SyncFlags(ctx context.Context, flags []domain.FeatureFlag) error {
	log := logger.FromContext(ctx)
	if r.writer == nil {
		log.Warn().Msg("Flipt writer not initialized, skipping flag sync")
		return nil
	}

	flagsMap := make(map[string]interface{})
	for _, flag := range flags {
		flagsMap[flag.Key] = flag.DefaultValue
	}

	return r.writer.SyncFlags(ctx, flagsMap)
}

func (r *FliptRepository) GetFlag(ctx context.Context, flagKey string) (*domain.FeatureFlag, error) {
	logger.FromContext(ctx).Warn().
		Str("flag_key", flagKey).
		Msg("GetFlag not implemented yet")
	return nil, fmt.Errorf("GetFlag not implemented")
}

var _ domain.FeatureFlagRepository = (*FliptRepository)(nil)
