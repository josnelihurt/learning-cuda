package connectrpc

import (
	"context"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/internal/config"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ConfigHandler struct {
	config *config.Config
}

func NewConfigHandler(cfg *config.Config) *ConfigHandler {
	return &ConfigHandler{
		config: cfg,
	}
}

func (h *ConfigHandler) GetStreamConfig(
	ctx context.Context,
	req *connect.Request[pb.GetStreamConfigRequest],
) (*connect.Response[pb.GetStreamConfigResponse], error) {
	span := trace.SpanFromContext(ctx)

	endpoints := []*pb.StreamEndpoint{
		{
			Type:            "websocket",
			Endpoint:        h.config.Stream.WebsocketEndpoint,
			TransportFormat: h.config.Stream.TransportFormat,
		},
	}

	span.SetAttributes(
		attribute.String("config.endpoint", h.config.Stream.WebsocketEndpoint),
		attribute.String("config.transport_format", h.config.Stream.TransportFormat),
		attribute.Int("config.endpoint_count", len(endpoints)),
	)

	return connect.NewResponse(&pb.GetStreamConfigResponse{
		Endpoints: endpoints,
	}), nil
}

