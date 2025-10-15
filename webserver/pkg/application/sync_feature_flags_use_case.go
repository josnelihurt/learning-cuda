package application

import (
	"context"
	"log"

	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type SyncFeatureFlagsUseCase struct {
	repository domain.FeatureFlagRepository
}

func NewSyncFeatureFlagsUseCase(repo domain.FeatureFlagRepository) *SyncFeatureFlagsUseCase {
	return &SyncFeatureFlagsUseCase{repository: repo}
}

func (uc *SyncFeatureFlagsUseCase) Execute(
	ctx context.Context,
	flags []domain.FeatureFlag,
) error {
	tracer := otel.Tracer("sync-feature-flags")
	ctx, span := tracer.Start(ctx, "SyncFeatureFlags",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(attribute.Int("flags.count", len(flags)))

	log.Printf("Syncing %d feature flags to repository", len(flags))

	err := uc.repository.SyncFlags(ctx, flags)
	if err != nil {
		span.RecordError(err)
		log.Printf("Failed to sync feature flags: %v", err)
		return err
	}

	log.Printf("Successfully synced %d feature flags", len(flags))
	return nil
}
