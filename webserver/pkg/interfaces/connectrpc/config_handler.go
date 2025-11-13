package connectrpc

import (
	"context"
	"fmt"
	"log"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/config"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
	"github.com/jrb/cuda-learning/webserver/pkg/infrastructure/processor"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ConfigHandler struct {
	getStreamConfigUseCase *application.GetStreamConfigUseCase
	syncFlagsUseCase       *application.SyncFeatureFlagsUseCase
	listInputsUseCase      *application.ListInputsUseCase
	evaluateFFUseCase      *application.EvaluateFeatureFlagUseCase
	getSystemInfoUseCase   *application.GetSystemInfoUseCase
	configManager          *config.Manager
	cppConnector           *processor.CppConnector
}

func NewConfigHandler(
	getStreamConfigUC *application.GetStreamConfigUseCase,
	syncFlagsUC *application.SyncFeatureFlagsUseCase,
	listInputsUC *application.ListInputsUseCase,
	evaluateFFUC *application.EvaluateFeatureFlagUseCase,
	getSystemInfoUC *application.GetSystemInfoUseCase,
	configManager *config.Manager,
	cppConnector *processor.CppConnector,
) *ConfigHandler {
	return &ConfigHandler{
		getStreamConfigUseCase: getStreamConfigUC,
		syncFlagsUseCase:       syncFlagsUC,
		listInputsUseCase:      listInputsUC,
		evaluateFFUseCase:      evaluateFFUC,
		getSystemInfoUseCase:   getSystemInfoUC,
		configManager:          configManager,
		cppConnector:           cppConnector,
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

	logLevel, err := h.evaluateFFUseCase.EvaluateVariant(ctx, "frontend_log_level", "default", "INFO")
	if err != nil || logLevel == "" {
		logLevel = "INFO"
	}

	consoleLogging, err := h.evaluateFFUseCase.EvaluateBoolean(ctx, "frontend_console_logging", "default", true)
	if err != nil {
		consoleLogging = true
	}

	// Force transport_format to "json" if empty
	transportFormat := streamConfig.TransportFormat
	if transportFormat == "" {
		transportFormat = "json"
		log.Printf("DEBUG: Forced transport_format to 'json' (was empty)")
	}

	endpoints := []*pb.StreamEndpoint{
		{
			Type:            "websocket",
			Endpoint:        streamConfig.WebsocketEndpoint,
			TransportFormat: transportFormat,
			LogLevel:        logLevel,
			ConsoleLogging:  consoleLogging,
		},
	}

	span.SetAttributes(
		attribute.String("config.endpoint", streamConfig.WebsocketEndpoint),
		attribute.String("config.transport_format", streamConfig.TransportFormat),
		attribute.String("config.log_level", logLevel),
		attribute.Bool("config.console_logging", consoleLogging),
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
		{
			Key:          "frontend_log_level",
			Name:         "Frontend Log Level",
			Type:         domain.VariantFlagType,
			Enabled:      true,
			DefaultValue: "INFO",
		},
		{
			Key:          "frontend_console_logging",
			Name:         "Frontend Console Logging",
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
			Id:               src.ID,
			DisplayName:      src.DisplayName,
			Type:             src.Type,
			ImagePath:        src.ImagePath,
			IsDefault:        src.IsDefault,
			VideoPath:        src.VideoPath,
			PreviewImagePath: src.PreviewImagePath,
		}
	}

	span.SetAttributes(
		attribute.Int("input_sources.count", len(pbSources)),
	)

	return connect.NewResponse(&pb.ListInputsResponse{
		Sources: pbSources,
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

	if len(h.configManager.Tools.Observability) > 0 {
		tools := h.buildTools(h.configManager.Tools.Observability, h.configManager.Environment)
		categories = append(categories, &pb.ToolCategory{
			Id:    "observability",
			Name:  "Observability",
			Tools: tools,
		})
	}

	if len(h.configManager.Tools.Features) > 0 {
		tools := h.buildTools(h.configManager.Tools.Features, h.configManager.Environment)
		categories = append(categories, &pb.ToolCategory{
			Id:    "features",
			Name:  "Features",
			Tools: tools,
		})
	}

	if len(h.configManager.Tools.Testing) > 0 {
		tools := h.buildTools(h.configManager.Tools.Testing, h.configManager.Environment)
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

func (h *ConfigHandler) GetSystemInfo(
	ctx context.Context,
	req *connect.Request[pb.GetSystemInfoRequest],
) (*connect.Response[pb.GetSystemInfoResponse], error) {
	span := trace.SpanFromContext(ctx)

	systemInfo, err := h.getSystemInfoUseCase.Execute(ctx)
	if err != nil {
		span.RecordError(err)
		log.Printf("Failed to get system info: %v", err)
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	// Map domain to proto
	response := &pb.GetSystemInfoResponse{
		Version: &pb.SystemVersion{
			GoVersion:    systemInfo.Version.GoVersion,
			CppVersion:   systemInfo.Version.CppVersion,
			ProtoVersion: systemInfo.Version.ProtoVersion,
			Branch:       systemInfo.Version.Branch,
			BuildTime:    systemInfo.Version.BuildTime,
			CommitHash:   systemInfo.Version.CommitHash,
		},
		Environment: systemInfo.Environment,
	}

	// Set span attributes
	span.SetAttributes(
		attribute.String("system.version.go", systemInfo.Version.GoVersion),
		attribute.String("system.version.cpp", systemInfo.Version.CppVersion),
		attribute.String("system.version.proto", systemInfo.Version.ProtoVersion),
		attribute.String("system.version.branch", systemInfo.Version.Branch),
		attribute.String("system.version.build_time", systemInfo.Version.BuildTime),
		attribute.String("system.version.commit_hash", systemInfo.Version.CommitHash),
		attribute.String("system.environment", systemInfo.Environment),
	)

	log.Printf("GetSystemInfo: returning system info for environment: %s, go_version: %s",
		systemInfo.Environment, systemInfo.Version.GoVersion)

	return connect.NewResponse(response), nil
}

func (h *ConfigHandler) GetProcessorStatus(
	ctx context.Context,
	req *connect.Request[pb.GetProcessorStatusRequest],
) (*connect.Response[pb.GetProcessorStatusResponse], error) {
	span := trace.SpanFromContext(ctx)

	if h.cppConnector == nil {
		span.RecordError(fmt.Errorf("C++ connector not available"))
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("processor not available"))
	}

	capabilities := h.cppConnector.GetCapabilities()
	apiVersion := h.cppConnector.GetAPIVersion()
	libraryVersion, err := h.cppConnector.GetLibraryVersion()
	if err != nil {
		libraryVersion = "unknown"
	}

	span.SetAttributes(
		attribute.String("processor.api_version", apiVersion),
		attribute.String("processor.library_version", libraryVersion),
	)

	if capabilities == nil {
		span.RecordError(fmt.Errorf("capabilities not available"))
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("processor capabilities not available"))
	}

	response := &pb.GetProcessorStatusResponse{
		ApiVersion:     apiVersion,
		Capabilities:   capabilities,
		CurrentLibrary: libraryVersion,
	}

	span.SetAttributes(
		attribute.Int("processor.filter_count", len(capabilities.Filters)),
	)

	log.Printf("GetProcessorStatus: returning capabilities with %d filters, api_version: %s, library_version: %s",
		len(capabilities.Filters), apiVersion, libraryVersion)

	return connect.NewResponse(response), nil
}
