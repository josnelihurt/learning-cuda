package adapters

import (
	pb "github.com/jrb/cuda-learning/proto/gen"
	"github.com/jrb/cuda-learning/webserver/pkg/domain"
)

type ProtobufAdapter struct{}

func NewProtobufAdapter() *ProtobufAdapter {
	return &ProtobufAdapter{}
}

func (a *ProtobufAdapter) ToFilters(pbFilters []pb.FilterType) []domain.FilterType {
	filters := make([]domain.FilterType, 0, len(pbFilters))
	for _, f := range pbFilters {
		switch f {
		case pb.FilterType_FILTER_TYPE_GRAYSCALE:
			filters = append(filters, domain.FilterGrayscale)
		case pb.FilterType_FILTER_TYPE_NONE:
			filters = append(filters, domain.FilterNone)
		}
	}
	return filters
}

func (a *ProtobufAdapter) ToAccelerator(pbAccel pb.AcceleratorType) domain.AcceleratorType {
	switch pbAccel {
	case pb.AcceleratorType_ACCELERATOR_TYPE_GPU:
		return domain.AcceleratorGPU
	case pb.AcceleratorType_ACCELERATOR_TYPE_CPU:
		return domain.AcceleratorCPU
	default:
		return domain.AcceleratorGPU
	}
}

func (a *ProtobufAdapter) ToGrayscaleType(pbType pb.GrayscaleType) domain.GrayscaleType {
	switch pbType {
	case pb.GrayscaleType_GRAYSCALE_TYPE_BT601:
		return domain.GrayscaleBT601
	case pb.GrayscaleType_GRAYSCALE_TYPE_BT709:
		return domain.GrayscaleBT709
	case pb.GrayscaleType_GRAYSCALE_TYPE_AVERAGE:
		return domain.GrayscaleAverage
	case pb.GrayscaleType_GRAYSCALE_TYPE_LIGHTNESS:
		return domain.GrayscaleLightness
	case pb.GrayscaleType_GRAYSCALE_TYPE_LUMINOSITY:
		return domain.GrayscaleLuminosity
	default:
		return domain.GrayscaleBT601
	}
}

func (a *ProtobufAdapter) ToDomainImage(req *pb.ProcessImageRequest) *domain.Image {
	return &domain.Image{
		Data:   req.ImageData,
		Width:  int(req.Width),
		Height: int(req.Height),
		Format: "raw",
	}
}

func (a *ProtobufAdapter) ToProtobufResponse(img *domain.Image) *pb.ProcessImageResponse {
	return &pb.ProcessImageResponse{
		Code:      0,
		Message:   "success",
		ImageData: img.Data,
		Width:     int32(img.Width),
		Height:    int32(img.Height),
		Channels:  int32(len(img.Data) / (img.Width * img.Height)),
	}
}

