package connectrpc

import (
	"context"
	"log"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ConfigHandler struct {
	streamConfig        config.StreamConfig
	fliptConfig         config.FliptConfig
	client              httpClient
	featureFlagsManager featureFlagsManager
}

func NewConfigHandler(streamCfg config.StreamConfig, fliptCfg config.FliptConfig, client httpClient, featureFlagsManager featureFlagsManager) *ConfigHandler {
	return &ConfigHandler{
		streamConfig:        streamCfg,
		fliptConfig:         fliptCfg,
		client:              client,
		featureFlagsManager: featureFlagsManager,
	}
}

func (h *ConfigHandler) GetStreamConfig(
	ctx context.Context,
	req *connect.Request[pb.GetStreamConfigRequest],
) (*connect.Response[pb.GetStreamConfigResponse], error) {
	span := trace.SpanFromContext(ctx)

	transportFormat := h.streamConfig.TransportFormat
	websocketEndpoint := h.streamConfig.WebsocketEndpoint

	endpoints := []*pb.StreamEndpoint{
		{
			Type:            "websocket",
			Endpoint:        websocketEndpoint,
			TransportFormat: transportFormat,
		},
	}

	span.SetAttributes(
		attribute.String("config.endpoint", websocketEndpoint),
		attribute.String("config.transport_format", transportFormat),
		attribute.Int("config.endpoint_count", len(endpoints)),
	)

	return connect.NewResponse(&pb.GetStreamConfigResponse{
		Endpoints: endpoints,
	}), nil
}

func (h *ConfigHandler) SyncFeatureFlags(
	ctx context.Context,
	req *connect.Request[pb.SyncFeatureFlagsRequest],
) (*connect.Response[pb.SyncFeatureFlagsResponse], error) {
	span := trace.SpanFromContext(ctx)

	flags := make(map[string]interface{})
	err := h.featureFlagsManager.Iterate(ctx, func(ctx context.Context, flagKey string, flagValue interface{}) error {
		flags[flagKey] = flagValue
		return nil
	})
	if err != nil {
		log.Printf("Failed to iterate feature flags: %v", err)
		if len(flags) == 0 {
			span.SetAttributes(attribute.String("sync.status", "error"))
			return connect.NewResponse(&pb.SyncFeatureFlagsResponse{
				Message: "No feature flags defined in YAML configuration",
			}), nil
		}
	}

	log.Printf("Manual flag sync triggered - syncing %d flags to Flipt", len(flags))
	span.SetAttributes(attribute.Int("sync.flag_count", len(flags)))

	writer := config.NewFliptWriter(h.fliptConfig.URL, h.fliptConfig.Namespace, h.client)
	err = writer.SyncFlags(ctx, flags)

	if err != nil {
		log.Printf("Flag sync failed: %v", err)
		span.SetAttributes(attribute.String("sync.status", "failed"))
		return connect.NewResponse(&pb.SyncFeatureFlagsResponse{
			Message: "Failed to sync flags to Flipt: " + err.Error(),
		}), nil
	}

	log.Println("Manual flag sync completed successfully")
	span.SetAttributes(attribute.String("sync.status", "success"))
	return connect.NewResponse(&pb.SyncFeatureFlagsResponse{
		Message: "Flags synced successfully to Flipt",
	}), nil
}
