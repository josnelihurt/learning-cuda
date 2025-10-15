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
	streamConfig config.StreamConfig
	client       httpClient
	featureFlags featureFlagsManager
}

func NewConfigHandler(streamCfg config.StreamConfig, featureFlagsManager featureFlagsManager, client httpClient) *ConfigHandler {
	return &ConfigHandler{
		streamConfig: streamCfg,
		client:       client,
		featureFlags: featureFlagsManager,
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

	err := h.featureFlags.Sync(ctx)
	if err != nil {
		log.Printf("Flag sync failed: %v", err)
		span.SetAttributes(
			attribute.String("sync.status", "failed"),
			attribute.String("error.message", err.Error()),
		)
		span.RecordError(err)

		return nil, connect.NewError(connect.CodeInternal, err)
	}

	log.Println("Manual flag sync completed successfully")
	span.SetAttributes(attribute.String("sync.status", "success"))
	return connect.NewResponse(&pb.SyncFeatureFlagsResponse{
		Message: "Flags synced successfully to Flipt",
	}), nil
}
