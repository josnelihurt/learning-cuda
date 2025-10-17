package connectrpc

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor/loader"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ConfigHandler struct {
	getStreamConfigUseCase *application.GetStreamConfigUseCase
	syncFlagsUseCase       *application.SyncFeatureFlagsUseCase
	listInputsUseCase      *application.ListInputsUseCase
	registry               *loader.Registry
	currentLoader          **loader.Loader
	loaderMutex            *sync.RWMutex
	configManager          *config.Manager
}

func NewConfigHandler(
	getStreamConfigUC *application.GetStreamConfigUseCase,
	syncFlagsUC *application.SyncFeatureFlagsUseCase,
	listInputsUC *application.ListInputsUseCase,
	registry *loader.Registry,
	currentLoader **loader.Loader,
	loaderMutex *sync.RWMutex,
	configManager *config.Manager,
) *ConfigHandler {
	return &ConfigHandler{
		getStreamConfigUseCase: getStreamConfigUC,
		syncFlagsUseCase:       syncFlagsUC,
		listInputsUseCase:      listInputsUC,
		registry:               registry,
		currentLoader:          currentLoader,
		loaderMutex:            loaderMutex,
		configManager:          configManager,
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

func (h *ConfigHandler) GetProcessorStatus(
	ctx context.Context,
	req *connect.Request[pb.GetProcessorStatusRequest],
) (*connect.Response[pb.GetProcessorStatusResponse], error) {
	span := trace.SpanFromContext(ctx)

	h.loaderMutex.RLock()
	currentLoader := *h.currentLoader
	h.loaderMutex.RUnlock()

	caps := currentLoader.CachedCapabilities()
	availableLibs := h.registry.ListVersions()

	allLibs := h.registry.GetAllLibraries()
	if _, hasMock := allLibs["mock"]; hasMock {
		availableLibs = append(availableLibs, "mock")
	}

	span.SetAttributes(
		attribute.String("processor.current_version", currentLoader.GetVersion()),
		attribute.Int("processor.available_count", len(availableLibs)),
	)

	return connect.NewResponse(&pb.GetProcessorStatusResponse{
		CurrentLibrary:     currentLoader.GetVersion(),
		ApiVersion:         loader.CurrentAPIVersion,
		Capabilities:       caps,
		AvailableLibraries: availableLibs,
	}), nil
}

func (h *ConfigHandler) ReloadProcessor(
	ctx context.Context,
	req *connect.Request[pb.ReloadProcessorRequest],
) (*connect.Response[pb.ReloadProcessorResponse], error) {
	span := trace.SpanFromContext(ctx)

	version := req.Msg.Version
	if version == "" {
		return nil, connect.NewError(connect.CodeInvalidArgument,
			fmt.Errorf("version is required"))
	}

	newLoader, err := h.registry.LoadLibrary(version)
	if err != nil {
		span.RecordError(err)
		return nil, connect.NewError(connect.CodeNotFound, err)
	}

	h.loaderMutex.Lock()
	oldLoader := *h.currentLoader
	*h.currentLoader = newLoader
	h.loaderMutex.Unlock()

	go func() {
		time.Sleep(5 * time.Second)
		if oldLoader != nil {
			oldLoader.Cleanup()
		}
	}()

	span.SetAttributes(
		attribute.String("processor.old_version", oldLoader.GetVersion()),
		attribute.String("processor.new_version", version),
	)

	log.Printf("Processor library reloaded: %s -> %s", oldLoader.GetVersion(), version)

	return connect.NewResponse(&pb.ReloadProcessorResponse{
		Status:  "success",
		Message: "Processor library reloaded to version " + version,
	}), nil
}

func (h *ConfigHandler) GetAvailableTools(
	ctx context.Context,
	req *connect.Request[pb.GetAvailableToolsRequest],
) (*connect.Response[pb.GetAvailableToolsResponse], error) {
	span := trace.SpanFromContext(ctx)

	if h.configManager == nil {
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("config manager not available"))
	}

	categories := []*pb.ToolCategory{}

	if len(h.configManager.ToolsConfig.Observability) > 0 {
		tools := h.buildTools(h.configManager.ToolsConfig.Observability, h.configManager.Environment)
		categories = append(categories, &pb.ToolCategory{
			Id:    "observability",
			Name:  "Observability",
			Tools: tools,
		})
	}

	if len(h.configManager.ToolsConfig.Features) > 0 {
		tools := h.buildTools(h.configManager.ToolsConfig.Features, h.configManager.Environment)
		categories = append(categories, &pb.ToolCategory{
			Id:    "features",
			Name:  "Features",
			Tools: tools,
		})
	}

	if len(h.configManager.ToolsConfig.Testing) > 0 {
		tools := h.buildTools(h.configManager.ToolsConfig.Testing, h.configManager.Environment)
		categories = append(categories, &pb.ToolCategory{
			Id:    "testing",
			Name:  "Testing",
			Tools: tools,
		})
	}

	span.SetAttributes(
		attribute.String("config.environment", h.configManager.Environment),
		attribute.Int("tools.category_count", len(categories)),
	)

	log.Printf("GetAvailableTools: returning %d categories for environment: %s",
		len(categories), h.configManager.Environment)

	return connect.NewResponse(&pb.GetAvailableToolsResponse{
		Categories: categories,
	}), nil
}

func (h *ConfigHandler) buildTools(toolDefs []config.ToolDefinition, environment string) []*pb.Tool {
	tools := make([]*pb.Tool, 0, len(toolDefs))

	for _, toolDef := range toolDefs {
		tool := &pb.Tool{
			Id:       toolDef.ID,
			Name:     toolDef.Name,
			IconPath: toolDef.IconPath,
			Type:     toolDef.Type,
		}

		if toolDef.Type == "url" {
			if environment == "production" {
				tool.Url = toolDef.URLProd
			} else {
				tool.Url = toolDef.URLDev
			}
		} else if toolDef.Type == "action" {
			tool.Action = toolDef.Action
		}

		tools = append(tools, tool)
	}

	return tools
}
