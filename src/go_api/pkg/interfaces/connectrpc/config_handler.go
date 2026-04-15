package connectrpc

import (
	"context"
	"fmt"
	"strconv"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/src/go_api/pkg/application"
	"github.com/jrb/cuda-learning/src/go_api/pkg/config"
	"github.com/jrb/cuda-learning/src/go_api/pkg/domain"
	"github.com/jrb/cuda-learning/src/go_api/pkg/infrastructure/logger"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ConfigHandler struct {
	featureFlagRepo      domain.FeatureFlagRepository
	listInputsUseCase    *application.ListInputsUseCase
	evaluateFFUseCase    *application.EvaluateFeatureFlagUseCase
	getSystemInfoUseCase *application.GetSystemInfoUseCase
	configManager        *config.Manager
	processorCapsUC      application.ProcessorCapabilitiesUseCase
}

// ConfigHandlerDeps groups all dependencies needed to create a ConfigHandler.
type ConfigHandlerDeps struct {
	FeatureFlagRepo domain.FeatureFlagRepository
	ListInputsUC    *application.ListInputsUseCase
	EvaluateFFUC    *application.EvaluateFeatureFlagUseCase
	GetSystemInfoUC *application.GetSystemInfoUseCase
	ConfigManager   *config.Manager
	ProcessorCapsUC application.ProcessorCapabilitiesUseCase
}

func NewConfigHandler(deps ConfigHandlerDeps) *ConfigHandler {
	return &ConfigHandler{
		featureFlagRepo:      deps.FeatureFlagRepo,
		listInputsUseCase:    deps.ListInputsUC,
		evaluateFFUseCase:    deps.EvaluateFFUC,
		getSystemInfoUseCase: deps.GetSystemInfoUC,
		configManager:        deps.ConfigManager,
		processorCapsUC:      deps.ProcessorCapsUC,
	}
}

func (h *ConfigHandler) GetStreamConfig(
	ctx context.Context,
	req *connect.Request[pb.GetStreamConfigRequest],
) (*connect.Response[pb.GetStreamConfigResponse], error) {
	span := trace.SpanFromContext(ctx)
	_ = req

	if h.configManager == nil {
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("config manager not available"))
	}

	if h.configManager.Server.WebRTCSignalingEndpoint == "" {
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("webrtc signaling endpoint not configured"))
	}

	logLevel, err := h.evaluateFFUseCase.EvaluateString(ctx, "frontend_log_level", "default", "INFO")
	if err != nil || logLevel == "" {
		logLevel = "INFO"
	}

	consoleLogging, err := h.evaluateFFUseCase.EvaluateBoolean(ctx, "frontend_console_logging", "default", true)
	if err != nil {
		consoleLogging = true
	}

	// Return WebRTC signaling endpoint via ConnectRPC
	endpoints := []*pb.StreamEndpoint{
		{
			Type:           "webrtc",
			Endpoint:       h.configManager.Server.WebRTCSignalingEndpoint,
			LogLevel:       logLevel,
			ConsoleLogging: consoleLogging,
		},
	}

	logger.FromContext(ctx).Debug().Bool("console_logging", consoleLogging).Msg("StreamEndpoint console_logging value")

	span.SetAttributes(
		attribute.String("config.endpoint", h.configManager.Server.WebRTCSignalingEndpoint),
		attribute.String("config.log_level", logLevel),
		attribute.Bool("config.console_logging", consoleLogging),
		attribute.Int("config.endpoint_count", len(endpoints)),
	)

	return connect.NewResponse(&pb.GetStreamConfigResponse{
		Endpoints: endpoints,
	}), nil
}

func (h *ConfigHandler) ListFeatureFlags(
	ctx context.Context,
	req *connect.Request[pb.ListFeatureFlagsRequest],
) (*connect.Response[pb.ListFeatureFlagsResponse], error) {
	_ = req
	if h.featureFlagRepo == nil {
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("feature flags repository not available"))
	}
	flags, err := h.featureFlagRepo.ListFlags(ctx)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	result := make([]*pb.ManagedFeatureFlag, 0, len(flags))
	for _, flag := range flags {
		result = append(result, &pb.ManagedFeatureFlag{
			Key:          flag.Key,
			Name:         flag.Name,
			Type:         string(flag.Type),
			Enabled:      flag.Enabled,
			DefaultValue: fmt.Sprintf("%v", flag.DefaultValue),
			Description:  flag.Description,
		})
	}
	return connect.NewResponse(&pb.ListFeatureFlagsResponse{Flags: result}), nil
}

func (h *ConfigHandler) UpsertFeatureFlag(
	ctx context.Context,
	req *connect.Request[pb.UpsertFeatureFlagRequest],
) (*connect.Response[pb.UpsertFeatureFlagResponse], error) {
	if req.Msg.GetFlag() == nil {
		return nil, connect.NewError(connect.CodeInvalidArgument, fmt.Errorf("flag is required"))
	}
	if h.featureFlagRepo == nil {
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("feature flags repository not available"))
	}
	in := req.Msg.GetFlag()
	flagType := domain.FeatureFlagType(in.GetType())
	defaultValue := interface{}(in.GetDefaultValue())
	if flagType == domain.BooleanFlagType {
		parsed, err := strconv.ParseBool(in.GetDefaultValue())
		if err != nil {
			return nil, connect.NewError(connect.CodeInvalidArgument, fmt.Errorf("invalid boolean default value: %w", err))
		}
		defaultValue = parsed
	}
	err := h.featureFlagRepo.UpsertFlag(ctx, domain.FeatureFlag{
		Key:          in.GetKey(),
		Name:         in.GetName(),
		Type:         flagType,
		Enabled:      in.GetEnabled(),
		DefaultValue: defaultValue,
		Description:  in.GetDescription(),
	})
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	return connect.NewResponse(&pb.UpsertFeatureFlagResponse{Message: "Flag updated successfully"}), nil
}

func (h *ConfigHandler) ListInputs(
	ctx context.Context,
	req *connect.Request[pb.ListInputsRequest],
) (*connect.Response[pb.ListInputsResponse], error) {
	span := trace.SpanFromContext(ctx)

	sources, err := h.listInputsUseCase.Execute(ctx)
	if err != nil {
		span.RecordError(err)
		logger.FromContext(ctx).Error().Err(err).Msg("Failed to list input sources")
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
		tools := h.buildTools(h.configManager.Tools.Observability)
		categories = append(categories, &pb.ToolCategory{
			Id:    "observability",
			Name:  "Observability",
			Tools: tools,
		})
	}

	if len(h.configManager.Tools.Features) > 0 {
		tools := h.buildTools(h.configManager.Tools.Features)
		categories = append(categories, &pb.ToolCategory{
			Id:    "features",
			Name:  "Features",
			Tools: tools,
		})
	}

	if len(h.configManager.Tools.Testing) > 0 {
		tools := h.buildTools(h.configManager.Tools.Testing)
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

	logger.FromContext(ctx).Debug().
		Int("category_count", len(categories)).
		Str("environment", h.configManager.Environment).
		Msg("GetAvailableTools: returning categories")

	return connect.NewResponse(&pb.GetAvailableToolsResponse{
		Categories: categories,
	}), nil
}

func (h *ConfigHandler) buildTools(toolDefs []config.ToolDefinition) []*pb.Tool {
	tools := make([]*pb.Tool, 0, len(toolDefs))

	for _, toolDef := range toolDefs {
		tool := &pb.Tool{
			Id:       toolDef.ID,
			Name:     toolDef.Name,
			IconPath: toolDef.IconPath,
			Type:     toolDef.Type,
		}

		if toolDef.Type == "url" {
			tool.Url = toolDef.URL
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
		logger.FromContext(ctx).Error().Err(err).Msg("Failed to get system info")
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

	logger.FromContext(ctx).Debug().
		Str("environment", systemInfo.Environment).
		Str("go_version", systemInfo.Version.GoVersion).
		Msg("GetSystemInfo: returning system info")

	return connect.NewResponse(response), nil
}

func (h *ConfigHandler) GetProcessorStatus(
	ctx context.Context,
	req *connect.Request[pb.GetProcessorStatusRequest],
) (*connect.Response[pb.GetProcessorStatusResponse], error) {
	span := trace.SpanFromContext(ctx)

	if h.processorCapsUC == nil {
		span.RecordError(fmt.Errorf("processor capabilities use case not available"))
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("processor not available"))
	}

	caps, origin, err := h.processorCapsUC.Execute(ctx, true) // force gRPC
	if err != nil {
		span.RecordError(err)
		return nil, connect.NewError(connect.CodeUnavailable, fmt.Errorf("failed to get processor capabilities: %w", err))
	}

	if caps == nil {
		span.RecordError(fmt.Errorf("capabilities not available"))
		return nil, connect.NewError(connect.CodeInternal, fmt.Errorf("processor capabilities not available"))
	}

	span.SetAttributes(
		attribute.String("processor.api_version", caps.ApiVersion),
		attribute.String("processor.backend_origin", string(origin)),
		attribute.Int("processor.filter_count", len(caps.Filters)),
	)

	response := &pb.GetProcessorStatusResponse{
		ApiVersion:     caps.ApiVersion,
		Capabilities:   caps,
		CurrentLibrary: caps.LibraryVersion,
	}

	logger.FromContext(ctx).Debug().
		Int("filter_count", len(caps.Filters)).
		Str("api_version", caps.ApiVersion).
		Str("library_version", caps.LibraryVersion).
		Str("origin", string(origin)).
		Msg("GetProcessorStatus: returning capabilities")

	return connect.NewResponse(response), nil
}
