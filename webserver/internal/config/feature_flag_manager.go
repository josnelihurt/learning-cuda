package config

import (
	"context"
	"log"

	flipt "go.flipt.io/flipt-client"
)

const (
	FeatureFlagTransportFormat      = "ws.transport_format"
	FeatureFlagObservabilityEnabled = "observability.enabled"
)

type FeatureFlagManager struct {
	reader                fliptClientInterface
	writer                *FliptWriter
	observabilityEnabled  bool
	streamTransportFormat string
}

func NewFeatureFlagManager(reader fliptClientInterface, writer *FliptWriter) *FeatureFlagManager {
	return &FeatureFlagManager{
		reader: reader,
		writer: writer,
	}
}

func (m *FeatureFlagManager) Sync(ctx context.Context) error {
	return m.writer.SyncFlags(ctx, map[string]interface{}{
		FeatureFlagTransportFormat:      m.streamTransportFormat,
		FeatureFlagObservabilityEnabled: m.observabilityEnabled,
	})
}

func (m *FeatureFlagManager) GetStreamTransportFormat(ctx context.Context) string {
	variant, err := m.reader.EvaluateString(ctx, &flipt.EvaluationRequest{
		FlagKey:  "enable_stream_transport_format",
		EntityID: "stream_transport_format",
	})
	if err != nil {
		log.Printf("Flipt evaluation failed for '%s': %v. Using YAML fallback.", "enable_stream_transport_format", err)
		return m.streamTransportFormat
	}
	return variant.VariantAttachment
}

func (m *FeatureFlagManager) IsObservabilityEnabled(ctx context.Context) bool {
	enabled, err := m.reader.EvaluateBoolean(ctx, &flipt.EvaluationRequest{
		FlagKey:  "enable_observability",
		EntityID: "observability",
	})
	if err != nil {
		log.Printf("Flipt evaluation failed for '%s': %v. Using YAML fallback.", "enable_observability", err)
		return m.observabilityEnabled
	}
	return enabled.Enabled
}
