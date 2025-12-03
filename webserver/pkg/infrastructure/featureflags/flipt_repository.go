package featureflags

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/logger"
	flipt "go.flipt.io/flipt-client"
)

// FliptWriterInterface defines the subset of FliptWriter methods used by the repository.
type FliptWriterInterface interface {
	SyncFlags(ctx context.Context, flags map[string]interface{}) error
	GetFlag(ctx context.Context, flagKey string) (*Flag, error)
}

type FliptRepository struct {
	reader FliptClientInterface
	writer FliptWriterInterface
}

func NewFliptRepository(reader FliptClientInterface, writer FliptWriterInterface) *FliptRepository {
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
	// Prefer HTTP flag state so that toggling via Flipt's API is reflected immediately.
	if r.writer != nil {
		if flag, err := r.writer.GetFlag(ctx, flagKey); err == nil && flag != nil {
			return &domain.FeatureFlagEvaluation{
				FlagKey:      flagKey,
				EntityID:     entityID,
				Result:       flag.Enabled,
				Success:      true,
				UsedFallback: false,
			}, nil
		}
	}

	// Fallback to gRPC-based evaluation when HTTP lookup fails.
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
	// Prefer HTTP flag state so that toggling via Flipt's API is reflected immediately.
	// For variant flags, try to get the first variant if no rules are configured.
	if r.writer != nil {
		if flag, err := r.writer.GetFlag(ctx, flagKey); err == nil && flag != nil {
			// If flag has variants, try to extract value from the first variant's attachment
			if len(flag.Variants) > 0 {
				var attachment map[string]interface{}
				if err := json.Unmarshal(flag.Variants[0].Attachment, &attachment); err == nil {
					if value, ok := attachment["value"].(string); ok && value != "" {
						return &domain.FeatureFlagEvaluation{
							FlagKey:      flagKey,
							EntityID:     entityID,
							Result:       value,
							Success:      true,
							UsedFallback: false,
						}, nil
					}
				}
				// If attachment parsing fails, use variant key as fallback
				if flag.Variants[0].Key != "" {
					return &domain.FeatureFlagEvaluation{
						FlagKey:      flagKey,
						EntityID:     entityID,
						Result:       flag.Variants[0].Key,
						Success:      true,
						UsedFallback: false,
					}, nil
				}
			}
		}
	}

	// Fallback to gRPC-based evaluation when HTTP lookup fails or no variants found.
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
