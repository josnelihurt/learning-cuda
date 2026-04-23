package connectrpc

import (
	"context"
	"errors"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	videoapp "github.com/jrb/cuda-learning/src/go_api/pkg/application/media/video"
	"github.com/jrb/cuda-learning/src/go_api/pkg/interfaces/adapters"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type ImageProcessorHandler struct {
	startVideoPlaybackUC useCase[videoapp.StartVideoPlaybackUseCaseInput, videoapp.StartVideoPlaybackUseCaseOutput]
	stopVideoPlaybackUC  useCase[videoapp.StopVideoPlaybackUseCaseInput, videoapp.StopVideoPlaybackUseCaseOutput]
	capabilities         processorCapabilitiesUseCase
	adapter              *adapters.ProtobufAdapter
	filterCodec          *adapters.FilterCodec
	grpcClient           interface {
		GetVersionInfo(context.Context, *pb.GetVersionInfoRequest) (*pb.GetVersionInfoResponse, error)
	}
}

func NewImageProcessorHandlerWithGRPC(
	capabilitiesUC processorCapabilitiesUseCase,
	startVideoPlaybackUC useCase[videoapp.StartVideoPlaybackUseCaseInput, videoapp.StartVideoPlaybackUseCaseOutput],
	stopVideoPlaybackUC useCase[videoapp.StopVideoPlaybackUseCaseInput, videoapp.StopVideoPlaybackUseCaseOutput],
	grpcClient interface {
		GetVersionInfo(context.Context, *pb.GetVersionInfoRequest) (*pb.GetVersionInfoResponse, error)
	},
) *ImageProcessorHandler {
	return &ImageProcessorHandler{
		startVideoPlaybackUC: startVideoPlaybackUC,
		stopVideoPlaybackUC:  stopVideoPlaybackUC,
		adapter:              adapters.NewProtobufAdapter(),
		filterCodec:          adapters.NewFilterCodec(),
		capabilities:         capabilitiesUC,
		grpcClient:           grpcClient,
	}
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
	if h.startVideoPlaybackUC == nil {
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("start video playback use case not configured"))
	}

	input := videoapp.StartVideoPlaybackUseCaseInput{
		VideoID:        req.Msg.GetVideoId(),
		SessionID:      req.Msg.GetSessionId(),
		Filters:        h.adapter.ToFilters(req.Msg.GetFilters()),
		Accelerator:    h.adapter.ToAccelerator(req.Msg.GetAccelerator()),
		GrayscaleType:  h.adapter.ToGrayscaleType(req.Msg.GetGrayscaleType()),
		BlurParams:     h.adapter.ToBlurParameters(req.Msg.GetBlurParams()),
		GenericFilters: req.Msg.GetGenericFilters(),
		ModelParams:    req.Msg.GetModelParams(),
		TraceContext:   req.Msg.GetTraceContext().GetTraceparent(),
		APIVersion:     req.Msg.GetApiVersion(),
	}

	result, err := h.startVideoPlaybackUC.Execute(ctx, input)
	if err != nil {
		return nil, connect.NewError(mapStreamVideoError(err), err)
	}

	resp := &pb.StartVideoPlaybackResponse{
		Code:         result.Code,
		Message:      result.Message,
		SessionId:    result.SessionID,
		TraceContext: &pb.TraceContext{Traceparent: result.TraceContext},
		ApiVersion:   result.APIVersion,
	}
	return connect.NewResponse(resp), nil
}

func (h *ImageProcessorHandler) StopVideoPlayback(
	ctx context.Context,
	req *connect.Request[pb.StopVideoPlaybackRequest],
) (*connect.Response[pb.StopVideoPlaybackResponse], error) {
	if h.stopVideoPlaybackUC == nil {
		return nil, connect.NewError(connect.CodeUnimplemented, errors.New("stop video playback use case not configured"))
	}

	input := videoapp.StopVideoPlaybackUseCaseInput{
		SessionID:    req.Msg.GetSessionId(),
		TraceContext: req.Msg.GetTraceContext().GetTraceparent(),
		APIVersion:   req.Msg.GetApiVersion(),
	}

	result, err := h.stopVideoPlaybackUC.Execute(ctx, input)
	if err != nil {
		return nil, connect.NewError(mapStreamVideoError(err), err)
	}

	resp := &pb.StopVideoPlaybackResponse{
		Code:         result.Code,
		Message:      result.Message,
		SessionId:    result.SessionID,
		Stopped:      result.Stopped,
		TraceContext: &pb.TraceContext{Traceparent: result.TraceContext},
		ApiVersion:   result.APIVersion,
	}
	return connect.NewResponse(resp), nil
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
	case errors.Is(err, videoapp.ErrVideoPlaybackMissingVideoID),
		errors.Is(err, videoapp.ErrVideoPlaybackMissingSession):
		return connect.CodeInvalidArgument
	case errors.Is(err, videoapp.ErrVideoPlaybackAlreadyRunning):
		return connect.CodeAlreadyExists
	case errors.Is(err, videoapp.ErrVideoPlaybackNotRunning):
		return connect.CodeNotFound
	default:
		return connect.CodeInternal
	}
}

var _ genconnect.ImageProcessorServiceHandler = (*ImageProcessorHandler)(nil)
