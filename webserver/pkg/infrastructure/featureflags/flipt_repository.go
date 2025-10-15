package featureflags

import (
	"context"
	"log"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
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

func (r *FliptRepository) EvaluateBoolean(
	ctx context.Context,
	flagKey, entityID string,
) (*domain.FeatureFlagEvaluation, error) {
	if r.reader == nil {
		log.Printf("Flipt client not initialized, returning fallback evaluation")
		return &domain.FeatureFlagEvaluation{
			FlagKey:      flagKey,
			EntityID:     entityID,
			Success:      false,
			UsedFallback: true,
		}, nil
	}

	resp, err := r.reader.EvaluateBoolean(ctx, &flipt.EvaluationRequest{
		FlagKey:  flagKey,
		EntityID: entityID,
	})

	if err != nil {
		log.Printf("Flipt boolean evaluation failed for '%s': %v", flagKey, err)
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
		Result:       resp.Enabled,
		Success:      true,
		UsedFallback: false,
	}, nil
}

func (r *FliptRepository) EvaluateVariant(
	ctx context.Context,
	flagKey, entityID string,
) (*domain.FeatureFlagEvaluation, error) {
	if r.reader == nil {
		log.Printf("Flipt client not initialized, returning fallback evaluation")
		return &domain.FeatureFlagEvaluation{
			FlagKey:      flagKey,
			EntityID:     entityID,
			Success:      false,
			UsedFallback: true,
		}, nil
	}

	resp, err := r.reader.EvaluateString(ctx, &flipt.EvaluationRequest{
		FlagKey:  flagKey,
		EntityID: entityID,
	})

	if err != nil {
		log.Printf("Flipt variant evaluation failed for '%s': %v", flagKey, err)
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
		Result:       resp.VariantAttachment,
		Success:      true,
		UsedFallback: false,
	}, nil
}

func (r *FliptRepository) SyncFlags(ctx context.Context, flags []domain.FeatureFlag) error {
	if r.writer == nil {
		log.Printf("Flipt writer not initialized, skipping flag sync")
		return nil
	}

	flagsMap := make(map[string]interface{})
	for _, flag := range flags {
		flagsMap[flag.Key] = flag.DefaultValue
	}

	return r.writer.SyncFlags(ctx, flagsMap)
}

func (r *FliptRepository) GetFlag(ctx context.Context, flagKey string) (*domain.FeatureFlag, error) {
	log.Printf("GetFlag not implemented yet for flag '%s'", flagKey)
	return nil, nil
}

var _ domain.FeatureFlagRepository = (*FliptRepository)(nil)
