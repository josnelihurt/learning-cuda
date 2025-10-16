package connectrpc

import (
	"context"
	"log"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ConfigHandler struct {
	getStreamConfigUseCase *application.GetStreamConfigUseCase
	syncFlagsUseCase       *application.SyncFeatureFlagsUseCase
	listInputsUseCase      *application.ListInputsUseCase
}

func NewConfigHandler(
	getStreamConfigUC *application.GetStreamConfigUseCase,
	syncFlagsUC *application.SyncFeatureFlagsUseCase,
	listInputsUC *application.ListInputsUseCase,
) *ConfigHandler {
	return &ConfigHandler{
		getStreamConfigUseCase: getStreamConfigUC,
		syncFlagsUseCase:       syncFlagsUC,
		listInputsUseCase:      listInputsUC,
	}
}

func (h *ConfigHandler) GetStreamConfig(
	ctx context.Context,
	req *connect.Request[pb.GetStreamConfigRequest],
) (*connect.Response[pb.GetStreamConfigResponse], error) {
	span := trace.SpanFromContext(ctx)

	streamConfig, err := h.getStreamConfigUseCase.Execute(ctx)
	if err != nil {
		span.RecordError(err)
		log.Printf("Failed to get stream config: %v", err)
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	endpoints := []*pb.StreamEndpoint{
		{
			Type:            "websocket",
			Endpoint:        streamConfig.WebsocketEndpoint,
			TransportFormat: streamConfig.TransportFormat,
		},
	}

	span.SetAttributes(
		attribute.String("config.endpoint", streamConfig.WebsocketEndpoint),
		attribute.String("config.transport_format", streamConfig.TransportFormat),
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

	flags := []domain.FeatureFlag{
		{
			Key:          "ws_transport_format",
			Name:         "WebSocket Transport Format",
			Type:         domain.VariantFlagType,
			Enabled:      true,
			DefaultValue: "json",
		},
		{
			Key:          "observability_enabled",
			Name:         "Observability Enabled",
			Type:         domain.BooleanFlagType,
			Enabled:      true,
			DefaultValue: true,
		},
	}

	err := h.syncFlagsUseCase.Execute(ctx, flags)
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

func (h *ConfigHandler) ListInputs(
	ctx context.Context,
	req *connect.Request[pb.ListInputsRequest],
) (*connect.Response[pb.ListInputsResponse], error) {
	span := trace.SpanFromContext(ctx)

	sources, err := h.listInputsUseCase.Execute(ctx)
	if err != nil {
		span.RecordError(err)
		log.Printf("Failed to list input sources: %v", err)
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	pbSources := make([]*pb.InputSource, len(sources))
	for i, src := range sources {
		pbSources[i] = &pb.InputSource{
			Id:          src.ID,
			DisplayName: src.DisplayName,
			Type:        src.Type,
			ImagePath:   src.ImagePath,
			IsDefault:   src.IsDefault,
		}
	}

	span.SetAttributes(
		attribute.Int("input_sources.count", len(pbSources)),
	)

	return connect.NewResponse(&pb.ListInputsResponse{
		Sources: pbSources,
	}), nil
}
