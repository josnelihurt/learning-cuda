package connectrpc

import (
	"context"

	"connectrpc.com/connect"
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/proto/gen/genconnect"
	"github.com/jrb/cuda-learning/webserver/internal/application"
	"github.com/jrb/cuda-learning/webserver/internal/domain"
)

type ImageProcessorHandler struct {
	useCase *application.ProcessImageUseCase
}

func NewImageProcessorHandler(useCase *application.ProcessImageUseCase) *ImageProcessorHandler {
	return &ImageProcessorHandler{
		useCase: useCase,
	}
}

func (h *ImageProcessorHandler) ProcessImage(
	ctx context.Context,
	req *connect.Request[pb.ProcessImageRequest],
) (*connect.Response[pb.ProcessImageResponse], error) {
	msg := req.Msg
	
	filters := make([]domain.FilterType, 0, len(msg.Filters))
	for _, f := range msg.Filters {
		switch f {
		case pb.FilterType_FILTER_TYPE_GRAYSCALE:
			filters = append(filters, domain.FilterGrayscale)
		case pb.FilterType_FILTER_TYPE_NONE:
			filters = append(filters, domain.FilterNone)
		}
	}
	
	var accelerator domain.AcceleratorType
	switch msg.Accelerator {
	case pb.AcceleratorType_ACCELERATOR_TYPE_GPU:
		accelerator = domain.AcceleratorGPU
	case pb.AcceleratorType_ACCELERATOR_TYPE_CPU:
		accelerator = domain.AcceleratorCPU
	default:
		accelerator = domain.AcceleratorGPU
	}
	
	var grayscaleType domain.GrayscaleType
	switch msg.GrayscaleType {
	case pb.GrayscaleType_GRAYSCALE_TYPE_BT601:
		grayscaleType = domain.GrayscaleBT601
	case pb.GrayscaleType_GRAYSCALE_TYPE_BT709:
		grayscaleType = domain.GrayscaleBT709
	case pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE:
		grayscaleType = domain.GrayscaleAverage
	case pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS:
		grayscaleType = domain.GrayscaleLightness
	case pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY:
		grayscaleType = domain.GrayscaleLuminosity
	default:
		grayscaleType = domain.GrayscaleBT601
	}
	
	domainImg := &domain.Image{
		Data:   msg.ImageData,
		Width:  int(msg.Width),
		Height: int(msg.Height),
		Format: "raw",
	}
	
	processedImg, err := h.useCase.Execute(domainImg, filters, accelerator, grayscaleType)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	
	resp := &pb.ProcessImageResponse{
		Code:      0,
		Message:   "success",
		ImageData: processedImg.Data,
		Width:     int32(processedImg.Width),
		Height:    int32(processedImg.Height),
		Channels:  int32(len(processedImg.Data) / (processedImg.Width * processedImg.Height)),
	}
	
	return connect.NewResponse(resp), nil
}

func (h *ImageProcessorHandler) StreamProcessVideo(
	ctx context.Context,
	stream *connect.BidiStream[pb.ProcessImageRequest, pb.ProcessImageResponse],
) error {
	// TODO: Implement video streaming using Connect-RPC bidirectional streaming
	return connect.NewError(connect.CodeUnimplemented, nil)
}

var _ genconnect.ImageProcessorServiceHandler = (*ImageProcessorHandler)(nil)

