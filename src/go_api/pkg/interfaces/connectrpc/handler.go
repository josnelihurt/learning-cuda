package connectrpc

import (
	"context"
	"errors"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/src/go_api/pkg/application"
	"github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/adapters"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ImageProcessorHandler struct {
	useCase       *application.ProcessImageUseCase
	streamVideoUC *application.StreamVideoUseCase
	adapter       *adapters.ProtobufAdapter
	filterCodec   *adapters.FilterCodec
	capabilities  application.ProcessorCapabilitiesUseCase
	evaluateFFUse *application.EvaluateFeatureFlagUseCase
	grpcClient    interface {
		GetVersionInfo(context.Context, *pb.GetVersionInfoRequest) (*pb.GetVersionInfoResponse, error)
	}
}

func NewImageProcessorHandlerWithGRPC(
	useCase *application.ProcessImageUseCase,
	capabilitiesUC application.ProcessorCapabilitiesUseCase,
	evaluateFFUse *application.EvaluateFeatureFlagUseCase,
	streamVideoUC *application.StreamVideoUseCase,
	grpcClient interface {
		GetVersionInfo(context.Context, *pb.GetVersionInfoRequest) (*pb.GetVersionInfoResponse, error)
	},
) *ImageProcessorHandler {
	return &ImageProcessorHandler{
		useCase:       useCase,
		streamVideoUC: streamVideoUC,
		adapter:       adapters.NewProtobufAdapter(),
		filterCodec:   adapters.NewFilterCodec(),
		capabilities:  capabilitiesUC,
		evaluateFFUse: evaluateFFUse,
		grpcClient:    grpcClient,
	}
}

func (h *ImageProcessorHandler) ProcessImage(
	ctx context.Context,
	req *connect.Request[pb.ProcessImageRequest],
) (*connect.Response[pb.ProcessImageResponse], error) {
	msg := req.Msg

	opts := h.adapter.ExtractProcessingOptions(msg)
	domainImg := h.adapter.ToDomainImage(msg)

	processedImg, err := h.useCase.Execute(ctx, domainImg, opts)
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

	useGRPC := h.grpcClient != nil
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

func (h *ImageProcessorHandler) StartVideoPlayback(
	ctx context.Context,
	req *connect.Request[pb.StartVideoPlaybackRequest],
) (*connect.Response[pb.StartVideoPlaybackResponse], error) {
	if h.streamVideoUC == nil {
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("stream video use case not configured"))
	}

	resp, err := h.streamVideoUC.Start(ctx, req.Msg)
	if err != nil {
		return nil, connect.NewError(mapStreamVideoError(err), err)
	}

	return connect.NewResponse(resp), nil
}

func (h *ImageProcessorHandler) StopVideoPlayback(
	ctx context.Context,
	req *connect.Request[pb.StopVideoPlaybackRequest],
) (*connect.Response[pb.StopVideoPlaybackResponse], error) {
	if h.streamVideoUC == nil {
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("stream video use case not configured"))
	}

	resp, err := h.streamVideoUC.Stop(ctx, req.Msg)
	if err != nil {
		return nil, connect.NewError(mapStreamVideoError(err), err)
	}

	return connect.NewResponse(resp), nil
}

func (h *ImageProcessorHandler) StreamProcessVideo(
	ctx context.Context,
	stream *connect.BidiStream[pb.ProcessImageRequest, pb.ProcessImageResponse],
) error {
	return connect.NewError(
		connect.CodeUnimplemented,
		errors.New("StreamProcessVideo is not implemented; use StartVideoPlayback/StopVideoPlayback"),
	)
}

func (h *ImageProcessorHandler) GetVersionInfo(
	ctx context.Context,
	req *connect.Request[pb.GetVersionInfoRequest],
) (*connect.Response[pb.GetVersionInfoResponse], error) {
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

func mapStreamVideoError(err error) connect.Code {
	switch {
	case errors.Is(err, application.ErrVideoPlaybackMissingVideoID),
		errors.Is(err, application.ErrVideoPlaybackMissingSession):
		return connect.CodeInvalidArgument
	case errors.Is(err, application.ErrVideoPlaybackAlreadyRunning):
		return connect.CodeAlreadyExists
	case errors.Is(err, application.ErrVideoPlaybackNotRunning):
		return connect.CodeNotFound
	default:
		return connect.CodeInternal
	}
}

var _ genconnect.ImageProcessorServiceHandler = (*ImageProcessorHandler)(nil)
