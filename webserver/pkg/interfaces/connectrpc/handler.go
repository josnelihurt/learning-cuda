package connectrpc

import (
	"context"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/pkg/application"
	"github.com/jrb/cuda-learning/webserver/pkg/interfaces/adapters"
)

type ImageProcessorHandler struct {
	useCase *application.ProcessImageUseCase
	adapter *adapters.ProtobufAdapter
}

func NewImageProcessorHandler(useCase *application.ProcessImageUseCase) *ImageProcessorHandler {
	return &ImageProcessorHandler{
		useCase: useCase,
		adapter: adapters.NewProtobufAdapter(),
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
