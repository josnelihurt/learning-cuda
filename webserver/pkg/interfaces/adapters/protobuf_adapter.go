package adapters

import (
	"log"

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
		case pb.FilterType_FILTER_TYPE_BLUR:
			filters = append(filters, domain.FilterBlur)
		case pb.FilterType_FILTER_TYPE_NONE:
			filters = append(filters, domain.FilterNone)
		case pb.FilterType_FILTER_TYPE_UNSPECIFIED:
			// Log warning for unspecified filter type
			log.Printf("Warning: received unspecified filter type, skipping")
		default:
			// Log warning for unknown filter type
			log.Printf("Warning: received unknown filter type %v, skipping", f)
		}
	}
	return filters
}

func (a *ProtobufAdapter) ToAccelerator(pbAccel pb.AcceleratorType) domain.AcceleratorType {
	switch pbAccel {
	case pb.AcceleratorType_ACCELERATOR_TYPE_CUDA:
		return domain.AcceleratorGPU
	case pb.AcceleratorType_ACCELERATOR_TYPE_CPU:
		return domain.AcceleratorCPU
	case pb.AcceleratorType_ACCELERATOR_TYPE_UNSPECIFIED:
		// Log warning for unspecified accelerator type
		log.Printf("Warning: received unspecified accelerator type, defaulting to GPU")
		return domain.AcceleratorGPU
	case pb.AcceleratorType_ACCELERATOR_TYPE_OPENCL:
		// Log warning for unsupported accelerator type
		log.Printf("Warning: received unsupported accelerator type OPENCL, defaulting to GPU")
		return domain.AcceleratorGPU
	default:
		// Log warning for unknown accelerator type
		log.Printf("Warning: received unknown accelerator type %v, defaulting to GPU", pbAccel)
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
	case pb.GrayscaleType_GRAYSCALE_TYPE_UNSPECIFIED:
		// Log warning for unspecified grayscale type
		log.Printf("Warning: received unspecified grayscale type, defaulting to BT601")
		return domain.GrayscaleBT601
	default:
		// Log warning for unknown grayscale type
		log.Printf("Warning: received unknown grayscale type %v, defaulting to BT601", pbType)
		return domain.GrayscaleBT601
	}
}

func (a *ProtobufAdapter) ToBorderMode(pbMode pb.BorderMode) domain.BorderMode {
	switch pbMode {
	case pb.BorderMode_BORDER_MODE_CLAMP:
		return domain.BorderModeClamp
	case pb.BorderMode_BORDER_MODE_REFLECT:
		return domain.BorderModeReflect
	case pb.BorderMode_BORDER_MODE_WRAP:
		return domain.BorderModeWrap
	case pb.BorderMode_BORDER_MODE_UNSPECIFIED:
		// Log warning for unspecified border mode
		log.Printf("Warning: received unspecified border mode, defaulting to REFLECT")
		return domain.BorderModeReflect
	default:
		// Log warning for unknown border mode
		log.Printf("Warning: received unknown border mode %v, defaulting to REFLECT", pbMode)
		return domain.BorderModeReflect
	}
}

func (a *ProtobufAdapter) ToBlurParameters(pbBlur *pb.GaussianBlurParameters) *domain.BlurParameters {
	if pbBlur == nil {
		return nil
	}

	return &domain.BlurParameters{
		KernelSize: pbBlur.KernelSize,
		Sigma:      pbBlur.Sigma,
		BorderMode: a.ToBorderMode(pbBlur.BorderMode),
		Separable:  pbBlur.Separable,
	}
}

func (a *ProtobufAdapter) ToProtobufBorderMode(mode domain.BorderMode) pb.BorderMode {
	switch mode {
	case domain.BorderModeClamp:
		return pb.BorderMode_BORDER_MODE_CLAMP
	case domain.BorderModeReflect:
		return pb.BorderMode_BORDER_MODE_REFLECT
	case domain.BorderModeWrap:
		return pb.BorderMode_BORDER_MODE_WRAP
	default:
		log.Printf("Warning: unknown border mode %v, defaulting to REFLECT", mode)
		return pb.BorderMode_BORDER_MODE_REFLECT
	}
}

func (a *ProtobufAdapter) ToProtobufBlurParameters(blur *domain.BlurParameters) *pb.GaussianBlurParameters {
	if blur == nil {
		return nil
	}

	return &pb.GaussianBlurParameters{
		KernelSize: blur.KernelSize,
		Sigma:      blur.Sigma,
		BorderMode: a.ToProtobufBorderMode(blur.BorderMode),
		Separable:  blur.Separable,
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
