package connectrpc

import (
	"context"
	"fmt"
	"strings"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/adapters"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type filterCapabilitiesProvider interface {
	GetCapabilities() *pb.LibraryCapabilities
}

type ImageProcessorHandler struct {
	useCase      *application.ProcessImageUseCase
	adapter      *adapters.ProtobufAdapter
	capabilities filterCapabilitiesProvider
}

func NewImageProcessorHandler(
	useCase *application.ProcessImageUseCase,
	capabilitiesProvider filterCapabilitiesProvider,
) *ImageProcessorHandler {
	return &ImageProcessorHandler{
		useCase:      useCase,
		adapter:      adapters.NewProtobufAdapter(),
		capabilities: capabilitiesProvider,
	}
}

func (h *ImageProcessorHandler) ProcessImage(
	ctx context.Context,
	req *connect.Request[pb.ProcessImageRequest],
) (*connect.Response[pb.ProcessImageResponse], error) {
	msg := req.Msg

	filters := h.adapter.ToFilters(msg.Filters)
	accelerator := h.adapter.ToAccelerator(msg.Accelerator)
	grayscaleType := h.adapter.ToGrayscaleType(msg.GrayscaleType)
	domainImg := h.adapter.ToDomainImage(msg)
	blurParams := h.adapter.ToBlurParameters(msg.BlurParams)

	processedImg, err := h.useCase.Execute(ctx, domainImg, filters, accelerator, grayscaleType, blurParams)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	resp := h.adapter.ToProtobufResponse(processedImg)

	return connect.NewResponse(resp), nil
}

func (h *ImageProcessorHandler) ListFilters(
	ctx context.Context,
	req *connect.Request[pb.ListFiltersRequest],
) (*connect.Response[pb.ListFiltersResponse], error) {
	span := trace.SpanFromContext(ctx)

	if h.capabilities == nil {
		err := fmt.Errorf("processor capabilities not available")
		span.RecordError(err)
		return nil, connect.NewError(connect.CodeUnavailable, err)
	}

	caps := h.capabilities.GetCapabilities()
	if caps == nil {
		err := fmt.Errorf("capabilities not available")
		span.RecordError(err)
		return nil, connect.NewError(connect.CodeUnavailable, err)
	}

	genericFilters := make([]*pb.GenericFilterDefinition, 0, len(caps.Filters))
	for _, def := range caps.Filters {
		if def == nil {
			continue
		}
		genericFilters = append(genericFilters, toGenericFilterDefinition(def))
	}

	span.SetAttributes(
		attribute.Int("filters.count", len(genericFilters)),
	)

	response := &pb.ListFiltersResponse{
		Filters:      genericFilters,
		ApiVersion:   caps.ApiVersion,
		TraceContext: req.Msg.GetTraceContext(),
	}

	return connect.NewResponse(response), nil
}

func (h *ImageProcessorHandler) StreamProcessVideo(
	ctx context.Context,
	stream *connect.BidiStream[pb.ProcessImageRequest, pb.ProcessImageResponse],
) error {
	// TODO: Implement this to replace WebSocket handler
	// This should replace: webserver/pkg/interfaces/websocket/handler.go HandleWebSocket method
	// Benefits: Type-safe streaming, unified protocol (Connect-RPC), better error handling
	// Implementation: Use stream.Receive() loop, call useCase.Execute, stream.Send() responses
	// Add tracing, handle context cancellation, manage backpressure
	return connect.NewError(connect.CodeUnimplemented, nil)
}

var _ genconnect.ImageProcessorServiceHandler = (*ImageProcessorHandler)(nil)

func toGenericFilterDefinition(def *pb.FilterDefinition) *pb.GenericFilterDefinition {
	if def == nil {
		return nil
	}

	parameters := make([]*pb.GenericFilterParameter, 0, len(def.Parameters))
	for _, param := range def.Parameters {
		parameters = append(parameters, toGenericFilterParameter(def.Id, param))
	}

	return &pb.GenericFilterDefinition{
		Id:                    def.Id,
		Name:                  def.Name,
		Parameters:            parameters,
		SupportedAccelerators: def.SupportedAccelerators,
	}
}

func toGenericFilterParameter(filterID string, param *pb.FilterParameter) *pb.GenericFilterParameter {
	if param == nil {
		return nil
	}

	options := make([]*pb.GenericFilterParameterOption, 0, len(param.Options))
	for _, option := range param.Options {
		options = append(options, &pb.GenericFilterParameterOption{
			Value: option,
			Label: formatParameterLabel(option),
		})
	}

	return &pb.GenericFilterParameter{
		Id:           param.Id,
		Name:         param.Name,
		Type:         mapParameterType(param.Type),
		Options:      options,
		DefaultValue: param.DefaultValue,
		Metadata:     buildParameterMetadata(filterID, param),
	}
}

func mapParameterType(paramType string) pb.GenericFilterParameterType {
	switch strings.ToLower(paramType) {
	case "select":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_SELECT
	case "range", "slider":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_RANGE
	case "number":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_NUMBER
	case "checkbox":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_CHECKBOX
	case "text", "string", "input":
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_TEXT
	default:
		return pb.GenericFilterParameterType_GENERIC_FILTER_PARAMETER_TYPE_UNSPECIFIED
	}
}

func buildParameterMetadata(filterID string, param *pb.FilterParameter) map[string]string {
	metadata := make(map[string]string)

	// Temporary heuristics for known parameters until metadata is provided by the accelerator.
	switch {
	case filterID == "blur" && param.Id == "kernel_size":
		metadata["min"] = "3"
		metadata["max"] = "15"
		metadata["step"] = "2"
	case filterID == "blur" && param.Id == "sigma":
		metadata["min"] = "0"
		metadata["max"] = "5"
		metadata["step"] = "0.1"
	case filterID == "blur" && param.Id == "border_mode":
		metadata["display"] = "select"
	case filterID == "blur" && param.Id == "separable":
		metadata["display"] = "checkbox"
	}

	if len(metadata) == 0 {
		return nil
	}
	return metadata
}

func formatParameterLabel(value string) string {
	switch strings.ToLower(value) {
	case "bt601":
		return "ITU-R BT.601 (SDTV)"
	case "bt709":
		return "ITU-R BT.709 (HDTV)"
	case "average":
		return "Average"
	case "lightness":
		return "Lightness"
	case "luminosity":
		return "Luminosity"
	default:
		return value
	}
}
