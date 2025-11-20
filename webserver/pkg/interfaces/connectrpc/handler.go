package connectrpc

import (
	"context"
	"fmt"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/adapters"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ImageProcessorHandler struct {
	useCase        *application.ProcessImageUseCase
	adapter        *adapters.ProtobufAdapter
	filterCodec    *adapters.FilterCodec
	capabilities   application.ProcessorCapabilitiesUseCase
	evaluateFFUse  *application.EvaluateFeatureFlagUseCase
	defaultUseGRPC bool
	grpcClient     interface {
		GetVersionInfo(context.Context, *pb.GetVersionInfoRequest) (*pb.GetVersionInfoResponse, error)
	}
}

func NewImageProcessorHandler(
	useCase *application.ProcessImageUseCase,
	capabilitiesUC application.ProcessorCapabilitiesUseCase,
	evaluateFFUse *application.EvaluateFeatureFlagUseCase,
	defaultUseGRPC bool,
) *ImageProcessorHandler {
	return &ImageProcessorHandler{
		useCase:        useCase,
		adapter:        adapters.NewProtobufAdapter(),
		filterCodec:    adapters.NewFilterCodec(),
		capabilities:   capabilitiesUC,
		evaluateFFUse:  evaluateFFUse,
		defaultUseGRPC: defaultUseGRPC,
	}
}

func NewImageProcessorHandlerWithGRPC(
	useCase *application.ProcessImageUseCase,
	capabilitiesUC application.ProcessorCapabilitiesUseCase,
	evaluateFFUse *application.EvaluateFeatureFlagUseCase,
	defaultUseGRPC bool,
	grpcClient interface {
		GetVersionInfo(context.Context, *pb.GetVersionInfoRequest) (*pb.GetVersionInfoResponse, error)
	},
) *ImageProcessorHandler {
	return &ImageProcessorHandler{
		useCase:        useCase,
		adapter:        adapters.NewProtobufAdapter(),
		filterCodec:    adapters.NewFilterCodec(),
		capabilities:   capabilitiesUC,
		evaluateFFUse:  evaluateFFUse,
		defaultUseGRPC: defaultUseGRPC,
		grpcClient:     grpcClient,
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

	useGRPC := h.defaultUseGRPC
	if h.evaluateFFUse != nil {
		value, err := h.evaluateFFUse.EvaluateBoolean(
			ctx,
			"processor_use_grpc_backend",
			"webserver",
			h.defaultUseGRPC,
		)
		if err == nil {
			useGRPC = value
		}
	}

	caps, origin, err := h.capabilities.Execute(ctx, useGRPC)
	if err != nil {
		span.RecordError(err)
		return nil, connect.NewError(connect.CodeUnavailable, err)
	}

	genericFilters := make([]*pb.GenericFilterDefinition, 0, len(caps.Filters))
	for _, def := range caps.Filters {
		if def == nil {
			continue
		}
		genericFilters = append(genericFilters, h.filterCodec.ToGenericFilterDefinition(def))
	}

	span.SetAttributes(
		attribute.Int("filters.count", len(genericFilters)),
	)

	response := &pb.ListFiltersResponse{
		Filters:      genericFilters,
		ApiVersion:   caps.ApiVersion,
		TraceContext: req.Msg.GetTraceContext(),
	}

	res := connect.NewResponse(response)
	res.Header().Set("X-Processor-Backend", string(origin))

	return res, nil
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

func (h *ImageProcessorHandler) GetVersionInfo(
	ctx context.Context,
	req *connect.Request[pb.GetVersionInfoRequest],
) (*connect.Response[pb.GetVersionInfoResponse], error) {
	if h.grpcClient == nil {
		return nil, connect.NewError(
			connect.CodeUnavailable,
			fmt.Errorf("gRPC client not available"),
		)
	}

	grpcReq := &pb.GetVersionInfoRequest{
		ApiVersion:   req.Msg.ApiVersion,
		TraceContext: req.Msg.TraceContext,
	}

	resp, err := h.grpcClient.GetVersionInfo(ctx, grpcReq)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}

	return connect.NewResponse(resp), nil
}

var _ genconnect.ImageProcessorServiceHandler = (*ImageProcessorHandler)(nil)
